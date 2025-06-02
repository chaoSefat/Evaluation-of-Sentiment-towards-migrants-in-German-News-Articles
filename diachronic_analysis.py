import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DiachronicSentimentAnalyzer:
    def __init__(self, data_directory):
        """
        Initialize the analyzer with data directory containing JSON files.
        Expected file naming: {language}_{year}.json
        """
        self.data_directory = Path(data_directory)
        self.data = {}
        self.sentiment_colors = {
            'positive': '#2E8B57',  # Sea Green
            'neutral': '#4682B4',   # Steel Blue
            'negative': '#DC143C',  # Crimson
            'irrelevant': '#D3D3D3' # Light Gray
        }
        
        # Key events for annotation
        self.key_events = {
            2010: "Haiti earthquake",
            2011: "Arab Spring begins",
            2013: "Syrian civil war escalates",
            2014: "Ukraine crisis",
            2015: "European refugee crisis peak",
            2016: "Brexit referendum",
            2017: "Trump travel ban",
            2018: "Global Compact for Migration",
            2019: "Central American migrant caravans",
            2020: "COVID-19 pandemic",
            2021: "Afghanistan withdrawal",
            2022: "Ukraine-Russia war",
            2023: "Mediterranean migration surge",
            2024: "US election year"
        }
        
    def load_data(self):
        """
        Load all JSON files from the data directory
        """
        print("📁 Loading data from JSON files...")
        
        for json_file in self.data_directory.glob("*.json"):
            try:
                # Parse filename to extract language and year
                filename = json_file.stem
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0] == 'sentiments':
                    language = parts[1]
                    year = int(parts[2])
                    
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if language not in self.data:
                        self.data[language] = {}
                    
                    self.data[language][year] = data
                    print(f"✅ Loaded {len(data)} sentences for {language} {year}")
                    
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")
        
        print(f"\n📊 Data loaded for languages: {list(self.data.keys())}")
        
    def compute_sentiment_distributions(self):
        """
        Compute sentiment distributions for each year and language
        """
        print("\n📊 Computing sentiment distributions...")
        
        self.distributions = {}
        self.raw_counts = {}
        
        for language in self.data:
            self.distributions[language] = {}
            self.raw_counts[language] = {}
            
            for year in sorted(self.data[language].keys()):
                sentences = self.data[language][year]
                
                # Count sentiments
                sentiment_counts = Counter([s['sentiment'] for s in sentences])
                total = len(sentences)
                
                # Compute proportions
                proportions = {
                    sentiment: (count / total) * 100 
                    for sentiment, count in sentiment_counts.items()
                }
                
                # Ensure all sentiment categories are present
                for sentiment in ['positive', 'neutral', 'negative', 'irrelevant']:
                    if sentiment not in proportions:
                        proportions[sentiment] = 0.0
                
                self.distributions[language][year] = proportions
                self.raw_counts[language][year] = dict(sentiment_counts)
                
                print(f"{year} ({language.upper()}): "
                      f"Positive: {proportions['positive']:.1f}%, "
                      f"Neutral: {proportions['neutral']:.1f}%, "
                      f"Negative: {proportions['negative']:.1f}%, "
                      f"Irrelevant: {proportions['irrelevant']:.1f}%")
    
    def create_time_series_visualization(self, apply_smoothing=True):
        """
        Create stacked area charts for sentiment evolution
        """
        print("\n📈 Creating time series visualizations...")
        
        fig, axes = plt.subplots(len(self.data), 1, figsize=(15, 6 * len(self.data)))
        if len(self.data) == 1:
            axes = [axes]
        
        for idx, language in enumerate(sorted(self.data.keys())):
            ax = axes[idx]
            
            years = sorted(self.distributions[language].keys())
            sentiments = ['positive', 'neutral', 'negative', 'irrelevant']
            
            # Prepare data for stacking
            data_matrix = np.zeros((len(sentiments), len(years)))
            for i, sentiment in enumerate(sentiments):
                for j, year in enumerate(years):
                    data_matrix[i, j] = self.distributions[language][year][sentiment]
            
            # Apply 3-year moving average if requested
            if apply_smoothing and len(years) > 3:
                smoothed_data = np.zeros_like(data_matrix)
                for i in range(len(sentiments)):
                    smoothed_data[i] = self._moving_average(data_matrix[i], window=3)
                data_matrix = smoothed_data
            
            # Create stacked area chart
            ax.stackplot(years, data_matrix, 
                        labels=sentiments,
                        colors=[self.sentiment_colors[s] for s in sentiments],
                        alpha=0.8)
            
            # Annotate key events
            for year in years:
                if year in self.key_events:
                    ax.axvline(x=year, color='red', linestyle='--', alpha=0.5)
                    ax.text(year, 95, self.key_events[year], 
                           rotation=90, ha='right', va='top', fontsize=8)
            
            ax.set_title(f'Sentiment Evolution - {language.upper()} News', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Percentage')
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sentiment_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_language_comparison(self):
        """
        Compare sentiment distributions between languages
        """
        print("\n🌍 Creating cross-language comparison...")
        
        if len(self.data) < 2:
            print("⚠️ Cross-language comparison requires at least 2 languages")
            return
        
        # Find common years
        languages = list(self.data.keys())
        common_years = set(self.data[languages[0]].keys())
        for lang in languages[1:]:
            common_years = common_years.intersection(set(self.data[lang].keys()))
        
        common_years = sorted(list(common_years))
        
        if not common_years:
            print("⚠️ No common years found between languages")
            return
        
        # Create side-by-side comparison
        sentiments = ['positive', 'neutral', 'negative', 'irrelevant']
        n_years = len(common_years)
        n_sentiments = len(sentiments)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, sentiment in enumerate(sentiments):
            ax = axes[i]
            
            # Prepare data for each language
            x = np.arange(len(common_years))
            width = 0.35
            
            lang1_data = [self.distributions[languages[0]][year][sentiment] for year in common_years]
            lang2_data = [self.distributions[languages[1]][year][sentiment] for year in common_years]
            
            ax.bar(x - width/2, lang1_data, width, label=languages[0].upper(), alpha=0.8)
            ax.bar(x + width/2, lang2_data, width, label=languages[1].upper(), alpha=0.8)
            
            ax.set_title(f'{sentiment.capitalize()} Sentiment Comparison', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Percentage')
            ax.set_xticks(x)
            ax.set_xticklabels([str(year) for year in common_years])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cross_language_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical comparison (Chi-squared tests)
        print("\n📐 Statistical comparison (Chi-squared tests):")
        chi_squared_results = {}
        
        for year in common_years:
            # Create contingency table
            lang1_counts = [self.raw_counts[languages[0]][year].get(s, 0) for s in sentiments]
            lang2_counts = [self.raw_counts[languages[1]][year].get(s, 0) for s in sentiments]
            
            contingency_table = np.array([lang1_counts, lang2_counts])
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                chi_squared_results[year] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{year}: χ² = {chi2:.3f}, p = {p_value:.4f} {significance}")
                
            except ValueError as e:
                print(f"{year}: Could not compute chi-squared test - {e}")
        
        return chi_squared_results
    
    def sentiment_volatility_analysis(self):
        """
        Analyze year-to-year changes in sentiment proportions
        """
        print("\n🧠 Analyzing sentiment volatility...")
        
        fig, axes = plt.subplots(len(self.data), 1, figsize=(15, 5 * len(self.data)))
        if len(self.data) == 1:
            axes = [axes]
        
        volatility_data = {}
        
        for idx, language in enumerate(sorted(self.data.keys())):
            ax = axes[idx]
            years = sorted(self.distributions[language].keys())
            sentiments = ['positive', 'neutral', 'negative', 'irrelevant']
            
            volatility_data[language] = {}
            
            for sentiment in sentiments:
                changes = []
                change_years = []
                
                for i in range(1, len(years)):
                    prev_year = years[i-1]
                    curr_year = years[i]
                    
                    prev_prop = self.distributions[language][prev_year][sentiment]
                    curr_prop = self.distributions[language][curr_year][sentiment]
                    
                    change = curr_prop - prev_prop
                    changes.append(change)
                    change_years.append(curr_year)
                
                volatility_data[language][sentiment] = {
                    'years': change_years,
                    'changes': changes
                }
                
                # Plot the changes
                ax.plot(change_years, changes, marker='o', label=sentiment, 
                       color=self.sentiment_colors[sentiment], linewidth=2)
            
            # Highlight years with major shifts
            for year in change_years:
                if year in self.key_events:
                    ax.axvline(x=year, color='red', linestyle='--', alpha=0.5)
                    ax.text(year, ax.get_ylim()[1]*0.9, self.key_events[year], 
                           rotation=90, ha='right', va='top', fontsize=8)
            
            ax.set_title(f'Sentiment Volatility - {language.upper()} News', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Change in Percentage Points')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sentiment_volatility.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print major volatility events
        print("\n📊 Major sentiment shifts identified:")
        for language in volatility_data:
            print(f"\n{language.upper()}:")
            for sentiment in volatility_data[language]:
                changes = volatility_data[language][sentiment]['changes']
                years = volatility_data[language][sentiment]['years']
                
                # Find largest changes
                if changes:
                    max_change_idx = np.argmax(np.abs(changes))
                    max_change = changes[max_change_idx]
                    max_change_year = years[max_change_idx]
                    
                    print(f"  {sentiment}: Largest change in {max_change_year} ({max_change:+.1f}pp)")
        
        return volatility_data
    
    def extract_representative_sentences(self, target_years=[2010, 2015, 2020, 2024]):
        """
        Extract representative sentences for selected years
        """
        print("\n📌 Extracting representative sentences...")
        
        representative_sentences = {}
        
        for language in self.data:
            representative_sentences[language] = {}
            
            for year in target_years:
                if year not in self.data[language]:
                    continue
                
                representative_sentences[language][year] = {}
                sentences = self.data[language][year]
                
                # Group sentences by sentiment
                sentiment_groups = defaultdict(list)
                for sentence in sentences:
                    sentiment_groups[sentence['sentiment']].append(sentence)
                
                for sentiment in ['positive', 'neutral', 'negative']:
                    if sentiment not in sentiment_groups:
                        continue
                    
                    group = sentiment_groups[sentiment]
                    
                    # Simple selection: pick sentences with diverse vocabulary
                    if len(group) >= 3:
                        # Sort by sentence length and pick diverse ones
                        sorted_sentences = sorted(group, key=lambda x: len(x['sentence']))
                        indices = [0, len(sorted_sentences)//2, -1]
                        selected = [sorted_sentences[i] for i in indices]
                    else:
                        selected = group
                    
                    representative_sentences[language][year][sentiment] = selected
        
        # Print representative sentences
        for language in representative_sentences:
            print(f"\n🌐 {language.upper()} Representative Sentences:")
            for year in sorted(representative_sentences[language].keys()):
                print(f"\n📅 {year}:")
                for sentiment in ['positive', 'neutral', 'negative']:
                    if sentiment in representative_sentences[language][year]:
                        print(f"\n  {sentiment.upper()}:")
                        for i, sentence_data in enumerate(representative_sentences[language][year][sentiment], 1):
                            print(f"    {i}. {sentence_data['sentence']}")
        
        return representative_sentences
    
    def linguistic_framing_analysis(self):
        """
        Analyze linguistic framing using keyword extraction
        """
        print("\n📐 Analyzing linguistic framing...")
        
        # Define framing keywords
        positive_keywords = [
            'welcome', 'support', 'aid', 'help', 'integration', 'opportunity', 
            'contribution', 'diversity', 'solidarity', 'protection', 'rescue',
            'humanitarian', 'shelter', 'assistance', 'hope'
        ]
        
        negative_keywords = [
            'burden', 'crisis', 'flood', 'wave', 'invasion', 'illegal', 'threat',
            'problem', 'strain', 'overwhelm', 'danger', 'risk', 'chaos',
            'abuse', 'exploit', 'criminal', 'security'
        ]
        
        framing_analysis = {}
        
        for language in self.data:
            framing_analysis[language] = {}
            
            for year in self.data[language]:
                sentences = [s['sentence'].lower() for s in self.data[language][year]]
                text_corpus = ' '.join(sentences)
                
                # Count framing keywords
                positive_count = sum(text_corpus.count(keyword) for keyword in positive_keywords)
                negative_count = sum(text_corpus.count(keyword) for keyword in negative_keywords)
                total_words = len(text_corpus.split())
                
                framing_analysis[language][year] = {
                    'positive_framing': (positive_count / total_words) * 1000,  # per 1000 words
                    'negative_framing': (negative_count / total_words) * 1000,
                    'framing_ratio': positive_count / max(negative_count, 1)
                }
        
        # Visualize framing trends
        fig, axes = plt.subplots(len(self.data), 1, figsize=(15, 5 * len(self.data)))
        if len(self.data) == 1:
            axes = [axes]
        
        for idx, language in enumerate(sorted(self.data.keys())):
            ax = axes[idx]
            years = sorted(framing_analysis[language].keys())
            
            positive_framing = [framing_analysis[language][year]['positive_framing'] for year in years]
            negative_framing = [framing_analysis[language][year]['negative_framing'] for year in years]
            
            ax.plot(years, positive_framing, marker='o', label='Positive Framing', 
                   color='green', linewidth=2)
            ax.plot(years, negative_framing, marker='s', label='Negative Framing', 
                   color='red', linewidth=2)
            
            ax.set_title(f'Linguistic Framing Trends - {language.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Keywords per 1000 words')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('linguistic_framing.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return framing_analysis
    
    def generate_statistical_summary(self):
        """
        Generate comprehensive statistical summary
        """
        print("\n🧮 Generating statistical summary...")
        
        summary_data = []
        
        for language in self.data:
            for year in sorted(self.data[language].keys()):
                row = {
                    'Language': language.upper(),
                    'Year': year,
                    'Total_Sentences': len(self.data[language][year])
                }
                
                # Add sentiment proportions
                for sentiment in ['positive', 'neutral', 'negative', 'irrelevant']:
                    row[f'{sentiment.capitalize()}_Percent'] = self.distributions[language][year][sentiment]
                    row[f'{sentiment.capitalize()}_Count'] = self.raw_counts[language][year].get(sentiment, 0)
                
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate statistics
        print("\n📊 Sentiment Statistics by Language:")
        for language in self.data:
            lang_data = summary_df[summary_df['Language'] == language.upper()]
            print(f"\n{language.upper()}:")
            
            for sentiment in ['Positive', 'Neutral', 'Negative', 'Irrelevant']:
                col = f'{sentiment}_Percent'
                if col in lang_data.columns:
                    mean_val = lang_data[col].mean()
                    std_val = lang_data[col].std()
                    print(f"  {sentiment}: Mean = {mean_val:.1f}% (±{std_val:.1f}%)")
        
        # Save summary table
        summary_df.to_csv('sentiment_summary.csv', index=False)
        print(f"\n💾 Summary table saved as 'sentiment_summary.csv'")
        
        return summary_df
    
    def _moving_average(self, data, window=3):
        """
        Apply moving average smoothing
        """
        if len(data) < window:
            return data
        
        smoothed = np.convolve(data, np.ones(window)/window, 'same')
        # Fix edge effects
        smoothed[:window//2] = data[:window//2]
        smoothed[-(window//2):] = data[-(window//2):]
        
        return smoothed
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("🚀 Starting complete diachronic sentiment analysis...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Compute distributions
        self.compute_sentiment_distributions()
        
        # 3. Time series visualization
        self.create_time_series_visualization()
        
        # 4. Cross-language comparison
        if len(self.data) > 1:
            chi_squared_results = self.cross_language_comparison()
        
        # 5. Volatility analysis
        volatility_data = self.sentiment_volatility_analysis()
        
        # 6. Representative sentences
        representative_sentences = self.extract_representative_sentences()
        
        # 7. Linguistic framing (optional)
        framing_analysis = self.linguistic_framing_analysis()
        
        # 8. Statistical summary
        summary_df = self.generate_statistical_summary()
        
        print("\n✅ Complete analysis finished!")
        print("📁 Generated files:")
        print("  - sentiment_time_series.png")
        print("  - cross_language_comparison.png")
        print("  - sentiment_volatility.png")
        print("  - linguistic_framing.png")
        print("  - sentiment_summary.csv")
        
        return {
            'distributions': self.distributions,
            'volatility': volatility_data,
            'representative_sentences': representative_sentences,
            'framing_analysis': framing_analysis,
            'summary': summary_df
        }

# Usage example:
if __name__ == "__main__":
    # Initialize analyzer with your data directory
    analyzer = DiachronicSentimentAnalyzer("output/prompt2")
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # You can also run individual analyses:
    # analyzer.load_data()
    # analyzer.compute_sentiment_distributions()
    # analyzer.create_time_series_visualization()
    # analyzer.cross_language_comparison()
    # analyzer.sentiment_volatility_analysis()
    # analyzer.extract_representative_sentences()
    # analyzer.linguistic_framing_analysis()
    # analyzer.generate_statistical_summary()