import sys
from src import train_spam_model
from src import train_gpt2
from src import train_instruction_fine_tuning
import src.llm1.train as llm1_train
import src.llm2.train as train_llm2

def main():

    valid_commands = {
        'help',
        "train_gpt2",
        "train_llm1",
        "train_llm2",
        "train_spam",
        "train_instruction"
    }
    command = sys.argv[1]
    assert command in valid_commands, f"Command '{command}' not found. Valid commands are: {valid_commands}"

    if command == 'help':
        print("Valid commands are: ")
        for comm in valid_commands:
            print(comm)

    if command == "train_gpt2":
        train_gpt2.run()
    
    if command == "train_llm1":
        llm1_train.run()
    
    if command == "train_llm2":
        train_llm2.run()
    
    if command == "train_spam":
        train_spam_model.run()

    if command == "train_instruction":
        train_instruction_fine_tuning.run()


if __name__ == "__main__":
    main()
