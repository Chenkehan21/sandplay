import logging
from datetime import datetime

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

import sys
sys.path.append('../')


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        logger
):
    output_dir = cfg.checkpoint.state_dict_dir
    model_name = cfg.model.name
    device = cfg.model.device
    epochs = cfg.train.epochs
    log_interval = cfg.train.log_interval
    tensorboard_dir = cfg.checkpoint.tensorboard_dir
    
    ignite_logger = logging.getLogger("ignite.engine.engine.Engine")
    ignite_logger.setLevel(logging.WARNING)
    logger.info("Start training")
    
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    
    metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(loss_fn)
    }
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        logger.info(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        logger.info(f"Training Results - Epoch[{engine.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        logger.info(f"Validation Results - Epoch[{engine.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    score_function = lambda engine: engine.state.metrics["accuracy"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpointer = ModelCheckpoint(dirname=f"{output_dir}/{timestamp}",
                                   filename_prefix='sandplay_'+model_name,
                                   n_saved=5,
                                   score_name="accuracy",
                                   score_function=score_function,
                                   require_empty=False)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"model": model, 'optimizer': optimizer})
    
    tb_logger = TensorboardLogger(log_dir=f"{tensorboard_dir}/{timestamp}")
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )
    
    trainer.run(train_loader, max_epochs=epochs)
    
    tb_logger.close()