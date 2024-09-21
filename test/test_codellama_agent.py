# Example usage
from codellama_agent import run_codellama_agent

if __name__ == "__main__":
    sample_code = """
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """
    result = run_codellama_agent(sample_code)
    print("Analysis:", result['analysis'])
    print("\nExplanation:", result['explanation'])
    print("\nSuggested Improvements:", result['improvements'])