import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
def calculate_overtalk(call_data):
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

def calculate_silence(call_data):
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

def visualize_metrics(call_data):
    """
    Create visualizations for call metrics
    """
    overtalk_pct = calculate_overtalk(call_data)
    silence_pct = calculate_silence(call_data)
    
    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Overtalk gauge - pie chart
    ax[0].pie([overtalk_pct, 100-overtalk_pct], 
              colors=['#ff9999', '#66b3ff'],
              labels=['Overtalk', 'Clean Speech'], 
              autopct='%1.1f%%',
              startangle=90)
    ax[0].set_title('Overtalk Percentage')
    
    # Silence gauge - pie chart
    ax[1].pie([silence_pct, 100-silence_pct], 
              colors=['#c2c2f0', '#99ff99'],
              labels=['Silence', 'Speech'], 
              autopct='%1.1f%%',
              startangle=90)
    ax[1].set_title('Silence Percentage')
    
    plt.tight_layout()
    st.pyplot(fig)

def visualize_timeline(call_data):
    """
    Visualize the call timeline showing who spoke when, 
    and highlighting overtalk and silence
    """
    # Convert to DataFrame
    df = pd.DataFrame(call_data)
    df['stime'] = pd.to_numeric(df['stime'])
    df['etime'] = pd.to_numeric(df['etime'])
    df['duration'] = df['etime'] - df['stime']
    
    # Sort by start time
    df = df.sort_values('stime')
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
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
    
    plt.tight_layout()
    st.pyplot(fig)

def analyze(call_data):
    """
    Analyze call quality metrics
    Returns: {
        "overtalk_percentage": float,
        "silence_percentage": float
    }
    """
    overtalk_pct = calculate_overtalk(call_data)
    silence_pct = calculate_silence(call_data)
    
    return {
        "overtalk_percentage": overtalk_pct,
        "silence_percentage": silence_pct
    }