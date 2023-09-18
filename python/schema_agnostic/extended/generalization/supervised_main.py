import logging
import os
from datetime import datetime
import json

from supervised_utils import build_optimizer, initialize_gpu_seed, load_data, DataType, \
                             DeepMatcherProcessor, train, predict,  setup_logging, \
                             read_arguments_train, write_config_to_file, \
                             Config, Evaluation, save_model
from time import time

setup_logging()

data_names=('dirty_amazon_itunes', 'abt_buy', 'dirty_walmart_amazon', 'dirty_dblp_acm', 'dirty_dblp_scholar')
max_lengths=(180, 265, 150, 180, 128)

def create_experiment_folder(model_output_dir: str, model_type: str, data_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}__{}".format(data_name.upper(), model_type.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == "__main__":
    args = read_arguments_train()

    exp_name = create_experiment_folder(args.model_output_dir, args.model_type, args.data_name)
    
    write_config_to_file(args, args.model_output_dir, exp_name)
    
    device, n_gpu = initialize_gpu_seed(args.seed)

    processor = DeepMatcherProcessor()

    label_list = processor.get_labels()

    logging.info("training with {} labels: {}".format(len(label_list), label_list))



    config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[args.model_type]
    if config_class is not None:
       config = config_class.from_pretrained(args.model_name_or_path)
       tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
       model = model_class.from_pretrained(args.model_name_or_path, config=config)
       model.to(device)
    else:       #SBERT Models
       tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
       model = model_class.from_pretrained(args.model_name_or_path)
       model.to(device)

    logging.info("initialized {}-model".format(args.model_type))

    args.data_path = os.path.join(args.data_dir, args.data_name)

    train_examples = processor.get_train_examples(args.data_path)


    training_data_loader = load_data(train_examples,
                                     label_list,
                                     tokenizer,
                                     args.max_seq_length,
                                     args.train_batch_size,
                                     DataType.TRAINING, args.model_type)
    logging.info("loaded {} training examples".format(len(train_examples)))

    num_train_steps = len(training_data_loader) * args.num_epochs

    optimizer, scheduler = build_optimizer(model,
                                           num_train_steps,
                                           args.learning_rate,
                                           args.adam_eps,
                                           args.warmup_steps,
                                           args.weight_decay)
    logging.info("Built optimizer: {}".format(optimizer))

    eval_examples = processor.get_dev_examples(args.data_path)
    evaluation_data_loader = load_data(eval_examples,
                                       label_list,
                                       tokenizer,
                                       args.max_seq_length,
                                       args.eval_batch_size,
                                       DataType.EVALUATION, args.model_type)

    evaluation = Evaluation(evaluation_data_loader, exp_name, args.model_output_dir, len(label_list), args.model_type)
    logging.info("loaded and initialized evaluation examples {}".format(len(eval_examples)))

    t1 = time()
    train(device,
          training_data_loader,
          model,
          optimizer,
          scheduler,
          evaluation,
          args.num_epochs,
          args.max_grad_norm,
          args.save_model_after_epoch,
          experiment_name=exp_name,
          output_dir=args.model_output_dir,
          model_type=args.model_type)
    t2 = time()
    training_time = t2-t1

    #Testing
    for data_name, max_length in zip(data_names, max_lengths):
        args.data_path = os.path.join(args.data_dir, data_name)
        test_examples = processor.get_test_examples(args.data_path)
    
        logging.info("loaded {} test examples".format(len(test_examples)))
        test_data_loader = load_data(test_examples,
                                     label_list,
                                     tokenizer,
                                     max_length,
                                     args.eval_batch_size,
                                     DataType.TEST, args.model_type)
    
        include_token_type_ids = False
        if args.model_type == 'bert':
           include_token_type_ids = True
           
        t1 = time()
        simple_accuracy, f1, classification_report, prfs, predictions = predict(model, device, test_data_loader, include_token_type_ids)
        t2 = time()
        testing_time = t2-t1
        logging.info("Prediction done for {} examples.F1: {}, Simple Accuracy: {}".format(len(test_data_loader), f1, simple_accuracy))
    
        logging.info(classification_report)
    
        print(predictions)
        #logging.info(predictions)         
        
        keys = ['precision', 'recall', 'fbeta_score', 'support']
        prfs = {f'class_{no}': {key: float(prfs[nok][no]) for nok, key in enumerate(keys)} for no in range(2)}
        
        log_file = args.log_dir + 'generalization.txt'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as fout:
            scores = {'simple_accuracy': simple_accuracy, 'f1': f1, 'model_type': args.model_type,
             'train_data_name': args.data_name, 'test_data_name': data_name, 
             'training_time': training_time, 'testing_time': testing_time, 'prfs': prfs}
            fout.write(json.dumps(scores)+"\n")
    
 
    '''     
    save_model(model, exp_name, args.model_output_dir, tokenizer=tokenizer)
    '''          
