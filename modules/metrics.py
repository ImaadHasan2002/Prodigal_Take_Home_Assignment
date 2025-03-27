import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from abc import ABC, abstractmethod

# SINGLE RESPONSIBILITY PRINCIPLE - Each class has just one responsibility

# INTERFACE SEGREGATION PRINCIPLE - Different interfaces for different metrics
class MetricsCalculator(ABC):
    @abstractmethod
    def calculate(self, call_data):
        pass

class MetricsVisualizer(ABC):
    @abstractmethod
    def visualize(self, call_data, figure=None):
        pass

# CONCRETE IMPLEMENTATIONS with Single Responsibility
class OvertalkMetricsCalculator(MetricsCalculator):
    def calculate(self, call_data):
        """
        Calculate percentage of call time with overtalk (simultaneous speaking)
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(call_data)
        df['stime'] = pd.to_numeric(df['stime'])
        df['etime'] = pd.to_numeric(df['etime'])
        
        # Sort by start time
        df = df.sort_values('stime')
        
        # Calculate total call duration
        call_start = df['stime'].min()
        call_end = df['etime'].max()
        call_duration = call_end - call_start
        
        # Find overlapping segments
        overtalk_duration = 0
        for i in range(len(df) - 1):
            current_end = df.iloc[i]['etime']
            next_start = df.iloc[i+1]['stime']
            next_speaker = df.iloc[i+1]['speaker']
            current_speaker = df.iloc[i]['speaker']
            
            # If speakers are different and there's overlap
            if current_speaker != next_speaker and current_end > next_start:
                overtalk_duration += min(current_end, df.iloc[i+1]['etime']) - next_start
        
        overtalk_percentage = (overtalk_duration / call_duration) * 100 if call_duration > 0 else 0
        return overtalk_percentage

class SilenceMetricsCalculator(MetricsCalculator):
    def calculate(self, call_data):
        """
        Calculate percentage of call time with silence (no one speaking)
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(call_data)
        df['stime'] = pd.to_numeric(df['stime'])
        df['etime'] = pd.to_numeric(df['etime'])
        
        # Sort by start time
        df = df.sort_values('stime')
        
        # Calculate total call duration
        call_start = df['stime'].min()
        call_end = df['etime'].max()
        call_duration = call_end - call_start
        
        # Merge overlapping segments
        merged_segments = []
        for start, end in zip(df['stime'], df['etime']):
            if not merged_segments or start > merged_segments[-1][1]:
                merged_segments.append([start, end])
            else:
                merged_segments[-1][1] = max(merged_segments[-1][1], end)
        
        # Calculate total speaking time
        speaking_duration = sum(end - start for start, end in merged_segments)
        
        # Calculate silence
        silence_duration = call_duration - speaking_duration
        silence_percentage = (silence_duration / call_duration) * 100 if call_duration > 0 else 0
        return silence_percentage

class SpeakerTimeMetricsCalculator(MetricsCalculator):
    """New metric calculator: speaking time per speaker"""
    def calculate(self, call_data):
        df = pd.DataFrame(call_data)
        df['stime'] = pd.to_numeric(df['stime'])
        df['etime'] = pd.to_numeric(df['etime'])
        df['duration'] = df['etime'] - df['stime']
        
        # Group by speaker and sum durations
        speaker_times = df.groupby('speaker')['duration'].sum().to_dict()
        
        # Calculate total call duration for percentages
        call_duration = df['etime'].max() - df['stime'].min()
        
        # Add percentages
        speaker_percentages = {speaker: (time/call_duration*100) for speaker, time in speaker_times.items()}
        
        return {
            "times": speaker_times,
            "percentages": speaker_percentages
        }

# VISUALIZATION IMPLEMENTATIONS
class PieChartVisualizer(MetricsVisualizer):
    def __init__(self, title, colors, labels):
        self.title = title
        self.colors = colors
        self.labels = labels
    
    def visualize(self, metric_value, figure=None):
        if figure is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig, ax = figure
        
        ax.pie([metric_value, 100-metric_value], 
               colors=self.colors,
               labels=self.labels, 
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title(self.title)
        
        return fig, ax

class TimelineVisualizer(MetricsVisualizer):
    def visualize(self, call_data, figure=None):
        # Convert to DataFrame
        df = pd.DataFrame(call_data)
        df['stime'] = pd.to_numeric(df['stime'])
        df['etime'] = pd.to_numeric(df['etime'])
        df['duration'] = df['etime'] - df['stime']
        
        # Sort by start time
        df = df.sort_values('stime')
        
        # Set up the figure
        if figure is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig, ax = figure
        
        # Calculate total call duration
        call_start = df['stime'].min()
        call_end = df['etime'].max()
        
        # Normalize times to start from 0
        df['stime_norm'] = df['stime'] - call_start
        df['etime_norm'] = df['etime'] - call_start
        
        # Plot agent and borrower utterances
        agent_color = '#66b3ff'  # Blue for agent
        borrower_color = '#ff9999'  # Red for borrower/customer
        
        # Plot silent periods
        current_time = 0
        for _, row in df.iterrows():
            if row['stime_norm'] > current_time:
                # There's silence
                ax.barh(0, row['stime_norm'] - current_time, left=current_time, 
                       height=0.8, color='#c2c2f0', alpha=0.5)
            current_time = max(current_time, row['etime_norm'])
        
        # If there's silence at the end
        if current_time < call_end - call_start:
            ax.barh(0, (call_end - call_start) - current_time, left=current_time, 
                   height=0.8, color='#c2c2f0', alpha=0.5)
        
        # Plot speakers
        for _, row in df.iterrows():
            y_pos = 1 if row['speaker'].lower() == 'agent' else 2
            ax.barh(y_pos, row['duration'], left=row['stime_norm'], 
                   height=0.8, 
                   color=agent_color if row['speaker'].lower() == 'agent' else borrower_color)
        
        # Add legend and labels
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Silence', 'Agent', 'Borrower'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Call Timeline')
        
        # Format the x-axis to show time in MM:SS format
        def format_time(x, pos):
            mins = int(x // 60)
            secs = int(x % 60)
            return f'{mins:02d}:{secs:02d}'
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        
        return fig, ax

# DEPENDENCY INVERSION - Factory for creating metrics calculators
class MetricsCalculatorFactory:
    @staticmethod
    def create(metric_type):
        if metric_type == "overtalk":
            return OvertalkMetricsCalculator()
        elif metric_type == "silence":
            return SilenceMetricsCalculator()
        elif metric_type == "speaker_time":
            return SpeakerTimeMetricsCalculator()
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

# OPEN/CLOSED PRINCIPLE - CallQualityAnalyzer can be extended without modification
class CallQualityAnalyzer:
    def __init__(self):
        # Initialize with default calculators, but can be extended
        self.calculators = {
            "overtalk": MetricsCalculatorFactory.create("overtalk"),
            "silence": MetricsCalculatorFactory.create("silence"),
            "speaker_time": MetricsCalculatorFactory.create("speaker_time")
        }
    
    def add_calculator(self, name, calculator):
        """Add a new calculator - OPEN FOR EXTENSION"""
        if not isinstance(calculator, MetricsCalculator):
            raise TypeError("Calculator must implement MetricsCalculator interface")
        self.calculators[name] = calculator
    
    def analyze(self, call_data, metrics=None):
        """
        Analyze call data using specified or all metrics
        """
        if metrics is None:
            metrics = list(self.calculators.keys())
        
        result = {}
        for metric in metrics:
            if metric in self.calculators:
                result[metric] = self.calculators[metric].calculate(call_data)
        
        return result

# Visualization coordinator (follows Single Responsibility)
class CallVisualizationManager:
    def __init__(self):
        self.visualizers = {
            "overtalk": PieChartVisualizer("Overtalk Percentage", 
                                           ['#ff9999', '#66b3ff'], 
                                           ['Overtalk', 'Clean Speech']),
            "silence": PieChartVisualizer("Silence Percentage", 
                                          ['#c2c2f0', '#99ff99'], 
                                          ['Silence', 'Speech']),
            "timeline": TimelineVisualizer()
        }
    
    def add_visualizer(self, name, visualizer):
        """Add a new visualizer - OPEN FOR EXTENSION"""
        if not isinstance(visualizer, MetricsVisualizer):
            raise TypeError("Visualizer must implement MetricsVisualizer interface")
        self.visualizers[name] = visualizer
    
    def visualize_metrics(self, call_data, analyzer):
        """Show metrics visualizations in Streamlit"""
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        # Calculate metrics
        metrics_results = analyzer.analyze(call_data)
        
        # Visualize overtalk
        self.visualizers["overtalk"].visualize(metrics_results["overtalk"], (fig, axs[0]))
        
        # Visualize silence
        self.visualizers["silence"].visualize(metrics_results["silence"], (fig, axs[1]))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Also show timeline separately
        fig, _ = self.visualizers["timeline"].visualize(call_data)
        st.pyplot(fig)
        
        # Show speaker time analysis
        self._visualize_speaker_time(metrics_results.get("speaker_time", None))
    
    def _visualize_speaker_time(self, speaker_time_data):
        if not speaker_time_data:
            return
            
        # Create a bar chart for speaker times
        fig, ax = plt.subplots(figsize=(8, 4))
        speakers = list(speaker_time_data["percentages"].keys())
        percentages = list(speaker_time_data["percentages"].values())
        
        bars = ax.bar(speakers, percentages, color=['#66b3ff', '#ff9999'])
        
        ax.set_ylabel('Speaking Time (%)')
        ax.set_title('Speaking Time Distribution')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        st.pyplot(fig)

# For backward compatibility - using the new architecture
def calculate_overtalk(call_data):
    calculator = OvertalkMetricsCalculator()
    return calculator.calculate(call_data)

def calculate_silence(call_data):
    calculator = SilenceMetricsCalculator()
    return calculator.calculate(call_data)

def visualize_metrics(call_data):
    analyzer = CallQualityAnalyzer()
    visualization_manager = CallVisualizationManager()
    visualization_manager.visualize_metrics(call_data, analyzer)

def visualize_timeline(call_data):
    visualizer = TimelineVisualizer()
    fig, _ = visualizer.visualize(call_data)
    st.pyplot(fig)

def analyze(call_data):
    """
    Analyze call quality metrics
    Returns: {
        "overtalk_percentage": float,
        "silence_percentage": float,
        "speaker_time": dict
    }
    """
    analyzer = CallQualityAnalyzer()
    results = analyzer.analyze(call_data)
    
    # For backward compatibility with existing code
    return {
        "overtalk_percentage": results["overtalk"],
        "silence_percentage": results["silence"],
        "speaker_time": results["speaker_time"]
    }