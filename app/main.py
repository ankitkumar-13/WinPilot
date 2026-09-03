from core.command_parser import CommandParser
from core.executor import ExecutionEngine


def main():

    parser = CommandParser()
    executor = ExecutionEngine()

    print("Welcome to WinPilot 0.0.1 !")
    print("Windows Desktop Automation Tool")
    print("================================")

    print("\nAvailable Commands:")
    print("1. open <application_name> - Opens the specified application.")
    print("2. create file <file_name> - Creates a new file.")
    print("3. create folder <folder_name> - Creates a new folder.")
    print("4. delete file <file_name> - Deletes the specified file.")
    print("5. delete folder <folder_name> - Deletes the specified folder.")
    print("6. search <name> - Searches for files and folders.")
    print("7. move <source> to <destination> - Moves a file or folder.")
    print("8. rename file <old_name> to <new_name> - Renames the specified file.")
    print("9. rename folder <old_name> to <new_name> - Renames the specified folder.")
    print("10. exit - Exits the application.")

    while True:
        print()
        user_input = input("WinPilot > ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Closing WinPilot...")
            break

        if not user_input:
            continue

        command = parser.parse_command(user_input)

        print(f"\nAction: {command.get('action')}")

        if command.get("target"):
            print(f"Target: {command.get('target')}")

        if command.get("source"):
            print(f"Source: {command.get('source')}")

        if command.get("destination"):
            print(f"Destination: {command.get('destination')}")

        if command.get("new_name"):
            print(f"New Name: {command.get('new_name')}")

        print("\nExecuting command...")

        success = executor.execute(command)

        if success:
            print("Command executed successfully.")
        else:
            print("Failed to execute command.")

if __name__ == "__main__":
    main()
