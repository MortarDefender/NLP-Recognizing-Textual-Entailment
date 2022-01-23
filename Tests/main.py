import sys
import json

sys.path.append('../')

from src.train import Trainer

# os.environ["WANDB_API_KEY"] = "0" ## to silence warning


def run(trainDataSet, testDataSet, outputFileName = None, verbose = False):
    if verbose:
        print(f"start training on {trainDataSet}")
    
    # classifier = Training(trainDataSet, testDataSet, verbose)
    classifier = Trainer(trainDataSet, verbose)
    
    if verbose:
        print("finished training")
    
    # classifier.predict()
    predictionResults = classifier.predict(testDataSet, outputFileName)
    
    if predictionResults is not None:
        print(predictionResults)
    
    print(classifier.accuracy(testDataSet))
    
    
    classifier = Trainer(
        ds_names={'original train': None, 'xnli valid': None, 'mnli train': 60000, 'mnli valid 1': None, 'mnli valid 2': None}, model_name=model_name,
        max_len=208, batch_size_per_replica=16, prediction_batch_size_per_replica=64,#16
        shuffle_buffer_size=None
    )
    
    print_config(trainer)
    
    train_name = f'{model_name} + extra-xnli-mnli'.replace('/', '-')
    history_3, submission_3,preds = trainer.train(train_name=train_name, model_name=model_name, epochs=epochs, verbose=True)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    run()

