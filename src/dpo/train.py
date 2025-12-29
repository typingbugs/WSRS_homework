from dpo.trainer import DPORankingTrainer
from dpo.model import init_model
from dpo.dataset import init_dataset
from dpo.metric import compute_metrics, preprocess_logits_for_metrics
from dpo.args import get_args
import logging


def init_logger():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )


def main():
    init_logger()
    logger = logging.getLogger(__name__)
    training_args, data_args, model_args = get_args()
    
    logger.info(f"Initializing model from {model_args.model_name_or_path}.")
    model, tokenizer = init_model(model_args)

    logger.info(f"Loading and processing data from {data_args.data_dir}.")
    train_set, validation_set = init_dataset(data_args, tokenizer)
    
    logger.info("Starting training...")
    trainer = DPORankingTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=validation_set,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    trainer.train()

    print(trainer.evaluate())

    logger.info("Training completed.")


if __name__ == "__main__":
    main()