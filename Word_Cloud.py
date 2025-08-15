from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Your text data of words and their frequencies
text_data_PU = """
text: 9
idea: 5
translation: 4
help: 4
coding: 4
word: 4
writing: 3
email: 3
work: 3
task: 3
research: 3
brainstorming: 3
topic: 3
information: 3
code: 3
formulation: 2
explanation: 2
search: 2
concept: 2
specific: 2
"""

text_data_IU = """
idea: 5
solve: 5
information: 4
research: 3
work: 3
task: 3
new: 3
know: 3
prompt: 3
help: 2
long: 2
time: 2
programming: 2
language: 2
image: 2
generation: 2
"""

text_data_CU = """
source: 4
information: 4
always: 3
code: 3
human: 2
problem: 2
come: 2
literature: 2
research: 2
hallucination: 2
using: 2
right: 2
wrong: 2
almost: 2
answer: 2
manipulate: 1
"""

# --- Step 1: Process BOTH data sources into separate dictionaries ---
frequencies_PU = {line.split(':')[0].strip(): int(line.split(':')[1].strip())
                  for line in text_data_PU.strip().split('\n')}

frequencies_IU = {line.split(':')[0].strip(): int(line.split(':')[1].strip())
                  for line in text_data_IU.strip().split('\n')}

frequencies_CU = {line.split(':')[0].strip(): int(line.split(':')[1].strip())
                  for line in text_data_CU.strip().split('\n')}


# --- Step 2: Generate and display the FIRST word cloud ---
wordcloud_PU = WordCloud(width=800,
                         height=400,
                         background_color="white",
                         colormap="viridis", # Using 'viridis' colormap
                         collocations=False).generate_from_frequencies(frequencies_PU)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_PU, interpolation='bilinear')
plt.title("Personal Use Cases (PU)") # Add a title
plt.axis("off")
plt.tight_layout(pad=0)


# --- Step 3: Generate and display the SECOND word cloud ---
wordcloud_IU = WordCloud(width=800,
                         height=400,
                         background_color="white",
                         colormap="plasma", # Using a different colormap 'plasma'
                         collocations=False).generate_from_frequencies(frequencies_IU)




plt.figure(figsize=(10, 5)) # Create a new figure for the second cloud
plt.imshow(wordcloud_IU, interpolation='bilinear')
plt.title("Impressive Use Cases (IU)") # Add a title
plt.axis("off")
plt.tight_layout(pad=0)


wordcloud_CU = WordCloud(width=800,
                         height=400,
                         background_color="white",
                         colormap="plasma",  # Using a different colormap 'plasma'
                         collocations=False).generate_from_frequencies(frequencies_CU)

plt.figure(figsize=(10, 5)) # Create a new figure for the second cloud
plt.imshow(wordcloud_CU, interpolation='bilinear')
plt.title("Concerning Use Cases (CU)") # Add a title
plt.axis("off")
plt.tight_layout(pad=0)

# --- Step 4: Show all generated figures ---
plt.show()