import json

def transform_json(input_filename="input.json", output_filename="output.json"):
    """
    Reads a JSON file with 'log' and 'action' keys, transforms it
    to a format with 'Instruction', 'Input', and 'Output' keys,
    and writes it to a new JSON file.

    Args:
        input_filename (str): The name of the input JSON file.
        output_filename (str): The name of the output JSON file.
    """
    instruction_text = "As a cybersecurity expert, suggest an action to mitigate the threat"
    transformed_data = []

    try:
        # Read the input JSON file
        with open(input_filename, 'r', encoding='utf-8') as infile:
            original_data = json.load(infile)

        # Ensure the input is a list (even if it's a list with one item)
        if not isinstance(original_data, list):
            # If it's a single dictionary, wrap it in a list
            if isinstance(original_data, dict):
                 original_data = [original_data]
            else:
                print(f"Error: Input file '{input_filename}' does not contain a valid JSON list or object.")
                return

        # Process each entry in the original data
        for entry in original_data:
            if "log" in entry and "action" in entry:
                transformed_entry = {
                    "Instruction": instruction_text,
                    "Input": entry["log"],
                    "Output": entry["action"]
                }
                transformed_data.append(transformed_entry)
            else:
                print(f"Warning: Skipping entry due to missing 'log' or 'action' key: {entry}")

        # Write the transformed data to the output JSON file
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            # Use indent for pretty printing, ensure_ascii=False for non-ASCII chars
            json.dump(transformed_data, outfile, indent=4, ensure_ascii=False)

        print(f"Successfully transformed '{input_filename}' to '{output_filename}'.")
        print(f"Total entries processed: {len(transformed_data)}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filename}'. Check if it's valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- How to Use ---
# 1. Save this code as a Python file (e.g., transform_logs.py).
# 2. Create an input JSON file named 'input.json' in the same directory.
#    Make sure it contains a JSON list of objects, like this example:
#    [
#      {
#        "log": "Malware detected in email attachment 'invoice.exe'.",
#        "action": "Quarantine email, delete attachment, and notify sender and recipient."
#      },
#      {
#        "log": "Unusual login attempt detected from IP 192.168.1.100 for user 'admin'.",
#        "action": "Block IP address, require multi-factor authentication for 'admin', and review account activity."
#      },
#      {
#        "log": "Potential phishing URL detected: http://example-bad-site.com",
#        "action": "Block URL at firewall/proxy, alert security team, and educate users about phishing."
#      }
#    ]
# 3. Run the script from your terminal: python transform_logs.py
# 4. A new file named 'output.json' will be created with the transformed data.

if __name__ == "__main__":
    # You can change the filenames here if needed
    input_file = "data/Final_dataset.json"
    output_file = "Dataset_Final.json"
    transform_json(input_file, output_file)