from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy


def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.model.device
    evaluator = create_supervised_evaluator(model, 
                                            metrics={'accuracy': Accuracy()}, 
                                            device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_acc = metrics['accuracy']
        print("Validation Results - Accuracy: {:.3f}".format(avg_acc))

    evaluator.run(val_loader)