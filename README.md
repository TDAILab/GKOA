# Graph-based Key Opinion Analyzer (GKOA)

## Overview

The Graph-based Key Opinion Analyzer (GKOA) is an innovative opinion mining framework designed to transform unstructured user-generated content (e.g., product reviews or social media comments) into an opinion graph using Large Language Models (LLMs). GKOA offers a comprehensive solution for analyzing complex online discussions, particularly useful for professionals such as journalists, policy makers, and social media analysts.

## Key Features

1. **Opinion Graph Construction**: Transforms text data into a structured graph format, enabling the use of global structure-based centrality measures and community detection algorithms.

2. **Multi-faceted Opinion Importance Evaluation**: Utilizes various centrality measures (PageRank, Betweenness, Closeness, Eigenvector, Degree) to quantify the importance of opinions from different perspectives.

3. **Summary Evaluation**: Accepts summaries created by humans or LLMs and identifies potentially overlooked important viewpoints, supporting human-centered summary creation and verification.

4. **Interactive Visualization**: Provides a network visualization of opinions, with color-coding and node sizes reflecting chosen centrality measures and inclusion in summaries.

## Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/GKOA.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Update the OpenAI API Key:
- Open the `settings.yaml` file
- Replace `<API KEY>` with your actual OpenAI API Key

## Usage
1. Run the Streamlit app:
```
streamlit run app.py
```

2. Use the web interface to:
- Upload CSV files or select from output folders for analysis
- Input or select summaries for evaluation
- Explore the opinion graph and analysis results

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or support, please contact [info@tdailab.com](mailto:info@tdailab.com).