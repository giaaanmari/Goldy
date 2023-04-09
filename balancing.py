import json

# Load the intents.json file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Count the number of patterns for each tag
counts = {}
for intent in intents['intents']:
    tag = tuple(intent['tag'])
    counts[tag] = counts.get(tag, 0) + len(intent['patterns'])

# Find the maximum number of patterns
max_patterns = max(counts.values())

# Balance the dataset by duplicating patterns
for intent in intents['intents']:
    tag = tuple(intent['tag'])
    patterns = intent['patterns']
    num_patterns = len(patterns)
    if num_patterns < max_patterns:
        diff = max_patterns - num_patterns
        for i in range(diff):
            pattern = patterns[i % num_patterns]
            intent['patterns'].append(pattern)

# Save the balanced dataset to a new file
with open('intents.json', 'w', encoding='utf-8') as file:
    json.dump(intents, file, indent=2, ensure_ascii=False)