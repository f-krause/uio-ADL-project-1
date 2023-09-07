# Group 06 - Project 1 

GitHub repository for the 1st Project of DL for Image Analysis Autumn 2023 at UiO. 

## Activate environment on educloud
Activate virtual environment on educloud
```shell
source /projects/ec232/venvs/in5310/bin/activate
```

## Run training from shell
Run full fine-tuning for 5 epochs from /src with
```shell
python main.py
```

To save the model after training specify the flag "-s" or "--save"
```shell
python main.py -s
```

To run LoRA training similarly use the "-l" or "--lora" flag

To specify a custom number of epochs use "-e" or "--epochs", run
```shell
python main.py -e 10
```
