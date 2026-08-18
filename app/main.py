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
    print("2. exit - Exits the application.")

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

        if command["target"]:
            print(f"Target: {command.get('target')}")

        print("\nExecuting command...")

        success = executor.execute(command)

        if success:
            print("Command executed successfully.")
        else:
            print("Failed to execute command.")

if __name__ == "__main__":
    main()
