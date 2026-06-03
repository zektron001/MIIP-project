from chat import ask


def main():
    print("MIIP Chat — type 'quit' to exit\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        answer = ask(question)
        print(f"\nGPT-4o: {answer}\n")


if __name__ == "__main__":
    main()
