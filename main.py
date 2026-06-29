import sys
import src.spam as spam
import src.instruction_fine_tune as finetune
import src.gpt2 as gpt2
import src.llm1 as llm1
import src.llm2 as llm2

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
        gpt2.train.run()
    
    if command == "train_llm1":
        llm1.train.run()
    
    if command == "train_llm2":
        llm2.train.run()
    
    if command == "train_spam":
        spam.train.run()

    if command == "train_instruction":
        finetune.train.run()


if __name__ == "__main__":
    main()
