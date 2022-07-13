from cvpods.engine.runner import (
    RUNNERS, torch, Infinite, hooks, comm, maybe_convert_module,
    DistributedDataParallel, auto_scale_config, DefaultCheckpointer, get_bn_modules
)

import time
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
import torch.nn.functional as F
from ema import ModelEMA
from loguru import logger

from cvpods.engine.runner import DefaultRunner
import numpy as np
from cvpods.modeling.losses import iou_loss
from dataset import build_train_loader

@RUNNERS.register()
class SemiRunner(DefaultRunner):
    def __init__(self, cfg, build_model):
        self.ema_start = cfg.TRAINER.EMA.START_STEPS
        
        self._hooks = []
        
        self.data_loader = build_train_loader(cfg)
        
        model = build_model(cfg)
        # Convert SyncBN to BatchNorm if only 1 GPU exists
        self.model = maybe_convert_module(model)
        logger.info(f"Model: \n{self.model}")

        self.optimizer = self.build_optimizer(cfg, self.model)

        # Don't modify code below unless you are expert
        if True:
            if cfg.TRAINER.FP16.ENABLED:
                self.mixed_precision = True
                if cfg.TRAINER.FP16.TYPE == "APEX":
                    from apex import amp
                    self.model, self.optimizer = amp.initialize(
                        self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                    )
            else:
                self.mixed_precision = False

            # For training, wrap with DDP. But don't need this for inference.
            if comm.get_world_size() > 1:
                torch.cuda.set_device(comm.get_local_rank())
                if cfg.MODEL.DDP_BACKEND == "torch":
                    self.model = DistributedDataParallel(
                        self.model,
                        device_ids=[comm.get_local_rank()],
                        broadcast_buffers=False,
                        find_unused_parameters=True
                    )
                elif cfg.MODEL.DDP_BACKEND == "apex":
                    from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                    self.model = ApexDistributedDataParallel(self.model)
                else:
                    raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))


            if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
                epoch_iters = -1
            else:
                epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
                logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

            auto_scale_config(cfg, self.data_loader)
        
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        self.model.train()
        self._data_loader_iter = iter(self.data_loader)
        

        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg

        self.decay_factor = cfg.TRAINER.EMA.DECAY_FACTOR
        self.burn_in_steps = cfg.TRAINER.SSL.BURN_IN_STEPS
        self.ema_update_steps = cfg.TRAINER.EMA.UPDATE_STEPS

        self.ema_model = ModelEMA(self.model, self.decay_factor)
        self.ema_model.model.eval()
        logger.info("EMA model built!")

        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema_model.model
        )

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg

        ret = [
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def test_and_save_ema_results():
            logger.info('Evaluating: EMA')
            results = self.test(self.cfg, self.ema_model.model)
            return results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_ema_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))
        return ret

    def resume_or_load(self, resume=True):
        self.checkpointer.resume = resume

        if not getattr(self.cfg.TRAINER.EMA, 'FAKE', False):
            # self.ema_checkpointer.resume = resume
            if resume:
                self.start_iter = (self.checkpointer.resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
            else:
                self.start_iter = (self.checkpointer.resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length

        self.scheduler.last_epoch = self.start_iter - 1

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[IterRunner] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self.epoch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
            
        unsup_weak, unsup_strong, sup_weak, sup_strong = [], [], [], []
        for d in data:
            uw, us, sw, ss = d
            unsup_weak.append(uw)
            unsup_strong.append(us)
            sup_weak.append(sw)
            sup_strong.append(ss)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """

        # Forward Logic
        sup_weak.extend(sup_strong)
        loss_dict_sup = self.model(sup_weak)

        loss_dict_sup = {k: v * self.cfg.TRAINER.DISTILL.SUP_WEIGHT for k,
                         v in loss_dict_sup.items()}
        losses_sup = sum([
            metrics_value for metrics_value in loss_dict_sup.values()
            if metrics_value.requires_grad
        ])
        losses_sup.backward()
        
        losses = losses_sup.detach()
        loss_dict = loss_dict_sup
        # Train Logic
        if self.iter > self.burn_in_steps:
            unsup_weight = self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT
            if self.cfg.TRAINER.DISTILL.SUPPRESS=='exp':
                target = self.burn_in_steps + 2000
                if self.iter <= target:
                    scale = np.exp((self.iter - target) / 1000)
                    unsup_weight *= scale
            elif self.cfg.TRAINER.DISTILL.SUPPRESS=='step':
                target = self.burn_in_steps * 2
                if self.iter <= target:
                    unsup_weight *= 0.25
            elif self.cfg.TRAINER.DISTILL.SUPPRESS=='linear':
                target = self.burn_in_steps * 2
                if self.iter <= target:
                    unsup_weight *= (self.iter-self.burn_in_steps)/self.burn_in_steps

                    
            student_logits, student_deltas, student_quality = self.model(unsup_strong, get_data=True)
            with torch.no_grad():
                teacher_logits, teacher_deltas, teacher_quality = self.ema_model.model(unsup_weak, is_teacher=True)
            loss_dict_unsup = self.get_distill_loss(student_logits, student_deltas, student_quality,
                                                    teacher_logits, teacher_deltas, teacher_quality)
            distill_weights = {
                "distill_loss_logits": self.cfg.TRAINER.DISTILL.WEIGHTS.LOGITS,
                "distill_loss_deltas": self.cfg.TRAINER.DISTILL.WEIGHTS.DELTAS,
                "distill_loss_quality": self.cfg.TRAINER.DISTILL.WEIGHTS.QUALITY,
                "fore_ground_sum": 1.,
            }
            loss_dict_unsup = {k: (v * unsup_weight) if v.requires_grad else v for k, v in loss_dict_unsup.items()}
            loss_dict_unsup = {k: v * distill_weights[k] for k, v in loss_dict_unsup.items()}
            losses_unsup = sum([
                metrics_value for metrics_value in loss_dict_unsup.values()
                if metrics_value.requires_grad
            ])
            losses_unsup.backward()
            loss_dict.update(loss_dict_unsup)
            losses += losses_unsup.detach()

        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Teacher Update Logic
        if self.iter == self.ema_start:
            # Start
            logger.info('EMA started.')
            self.ema_model.update(self.model, decay=0.)
        elif self.iter > self.ema_start and (self.iter - self.ema_start) % self.ema_update_steps == 0:
            self.ema_model.update(self.model)

        self._detect_anomaly(losses, loss_dict)
        self._write_metrics(loss_dict, data_time)

        self.step_outputs = {
            "loss_for_backward": losses,
        }

        self.inner_iter += 1

    def get_distill_loss(self, 
                         student_logits, student_deltas, student_quality, 
                         teacher_logits, teacher_deltas, teacher_quality):
        num_classes = self.cfg.MODEL.FCOS.NUM_CLASSES

        student_logits = torch.cat([
            permute_to_N_HWA_K(x, num_classes) for x in student_logits
        ], dim=1).view(-1, num_classes)
        teacher_logits = torch.cat([
            permute_to_N_HWA_K(x, num_classes) for x in teacher_logits
        ], dim=1).view(-1, num_classes)
        
        student_deltas = torch.cat([
            permute_to_N_HWA_K(x, 4) for x in student_deltas
        ], dim=1).view(-1, 4)
        teacher_deltas = torch.cat([
            permute_to_N_HWA_K(x, 4) for x in teacher_deltas
        ], dim=1).view(-1, 4)

        student_quality = torch.cat([
            permute_to_N_HWA_K(x, 1) for x in student_quality
        ], dim=1).view(-1, 1)
        teacher_quality = torch.cat([
            permute_to_N_HWA_K(x, 1) for x in teacher_quality
        ], dim=1).view(-1, 1)

        with torch.no_grad():
            # Region Selection
            ratio = self.cfg.TRAINER.DISTILL.RATIO
            count_num = int(teacher_logits.size(0) * ratio)
            teacher_probs = teacher_logits.sigmoid()
            max_vals = torch.max(teacher_probs, 1)[0]
            sorted_vals, sorted_inds = torch.topk(max_vals, teacher_logits.size(0))
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask=mask>0.
            
        loss_logits = QFLv2(
            student_logits.sigmoid(),
            teacher_probs,
            weight=mask,
            reduction="sum",
        ) / fg_num

        loss_deltas = (iou_loss(
            student_deltas[b_mask],
            teacher_deltas[b_mask],
            box_mode="ltrb",
            loss_type='giou',
            reduction="none",
        ) * teacher_quality[b_mask]).mean()

        loss_quality = F.binary_cross_entropy(
            student_quality[b_mask].sigmoid(),
            teacher_quality[b_mask].sigmoid(),
            reduction='mean'
        )

        return {
            "distill_loss_logits": loss_logits,
            "distill_loss_quality": loss_quality,
            "distill_loss_deltas": loss_deltas,
            "fore_ground_sum": fg_num,
        }


def QFLv2(pred_sigmoid,          # (n, 80)
          teacher_sigmoid,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss
