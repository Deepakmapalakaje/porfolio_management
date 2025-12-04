import csv

def search_nse_data():
    """
    Searches for data in the NSE.csv file based on user input.
    """
    while True:
        search_term = input("Enter the 'name' or 'tradingsymbol' to search (or 'exit' to quit): ").strip().lower()

        if search_term == 'exit':
            break

        if len(search_term) < 2:
            print("Please enter at least 2 characters to search.")
            continue

        suggestions = set()
        results = []
        
        try:
            with open('NSE.csv', 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get('instrument_type') == 'EQUITY':
                        name = row.get('name', '').lower()
                        tradingsymbol = row.get('tradingsymbol', '').lower()

                        if search_term in name or search_term in tradingsymbol:
                            results.append(row)
                        
                        if name.startswith(search_term):
                            suggestions.add(row['name'])
                        if tradingsymbol.startswith(search_term):
                            suggestions.add(row['tradingsymbol'])

        except FileNotFoundError:
            print("Error: NSE.csv not found. Please make sure the file is in the correct directory.")
            return

        if results:
            print(f"Found {len(results)} match(es) for '{search_term}':")
            for row in results:
                print(row)
        elif suggestions:
            print("No exact matches found. Did you mean one of these?")
            for item in sorted(list(suggestions))[:10]:  # Limit suggestions
                print(f"- {item}")
        else:
            print(f"No match found for '{search_term}'.")

if __name__ == "__main__":
    search_nse_data()