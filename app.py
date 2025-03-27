import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from modules import profanity, privacy, metrics
from utils import loader

def main():
    st.title("Debt Collection Call Analyzer")
    st.subheader("Analyze calls for compliance, profanity, and metrics")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload call file (YAML or JSON)", type=["json", "yaml", "yml"])
    
    if uploaded_file:
        try:
            # Parse the file
            call_data = loader.parse_file(uploaded_file)
            
            # Display basic call info
            st.markdown(f"## Call ID: {uploaded_file.name.split('.')[0]}")
            st.markdown(f"**Total Utterances:** {len(call_data)}")
            
            # Analysis options
            analysis_type = st.sidebar.selectbox(
                "Analysis Type", 
                ["Profanity Detection", "Privacy & Compliance", "Call Quality Metrics"]
            )
            
            if analysis_type in ["Profanity Detection", "Privacy & Compliance"]:
                approach = st.sidebar.selectbox(
                    "Analysis Approach", 
                    ["Pattern Matching (Regex)", "Machine Learning Approach"]
                )
                
                approach_param = "Regex" if "Regex" in approach else "AI"
                
                # Run analysis based on type
                if analysis_type == "Profanity Detection":
                    with st.spinner("Analyzing for profanity..."):
                        result = profanity.analyze(call_data, approach_param)
                    
                    # Display results
                    st.subheader("Profanity Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Agent")
                        if result["agent_profanity"]:
                            st.error("⚠️ Agent used profane language")
                            
                            # Show offending utterances
                            agent_utterances = [u for u in call_data if u["speaker"].lower() == "agent" and 
                                            profanity.check_profanity_in_text(u["text"], approach_param)]
                            
                            if agent_utterances:
                                st.markdown("#### Offending utterances:")
                                for utterance in agent_utterances:
                                    st.markdown(f"*{utterance['text']}*")
                        else:
                            st.success("✅ No profanity detected from agent")
                        
                    with col2:
                        st.markdown("### Borrower")
                        if result["borrower_profanity"]:
                            st.warning("⚠️ Borrower used profane language")
                            
                            # Show offending utterances
                            borrower_utterances = [u for u in call_data if u["speaker"].lower() == "borrower" and 
                                               profanity.check_profanity_in_text(u["text"], approach_param)]
                            
                            if borrower_utterances:
                                st.markdown("#### Offending utterances:")
                                for utterance in borrower_utterances:
                                    st.markdown(f"*{utterance['text']}*")
                        else:
                            st.info("✅ No profanity detected from borrower")
                
                elif analysis_type == "Privacy & Compliance":
                    with st.spinner("Analyzing for compliance violations..."):
                        result = privacy.analyze(call_data, approach_param)
                    
                    # Display results
                    st.subheader("Privacy & Compliance Analysis")
                    
                    if result["privacy_violation"]:
                        st.error("⚠️ COMPLIANCE VIOLATION: Sensitive information shared without verification")
                        
                        # Show verification status
                        verification_data = privacy.get_verification_status(call_data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Verification Status")
                            st.markdown(f"Identity Verified: {'✅ Yes' if verification_data['verified'] else '❌ No'}")
                        
                        with col2:
                            st.markdown("### Sensitive Information")
                            for info_type, was_shared in verification_data["sensitive_info_shared"].items():
                                status = "⚠️ Shared" if was_shared else "Not shared"
                                st.markdown(f"{info_type}: {status}")
                        
                        # Show problematic utterances
                        if verification_data["problematic_utterances"]:
                            st.markdown("### Problematic Utterances")
                            for utterance in verification_data["problematic_utterances"]:
                                st.markdown(f"*{utterance['text']}*")
                    else:
                        st.success("✅ No privacy violations detected")
                        
                    # Display verification process info
                    with st.expander("View Verification Process"):
                        verification_data = privacy.get_verification_status(call_data)
                        
                        if verification_data["verification_utterances"]:
                            st.markdown("### Verification Process")
                            for utterance in verification_data["verification_utterances"]:
                                st.markdown(f"*{utterance['text']}*")
                        else:
                            st.markdown("No verification attempts detected")
            
            elif analysis_type == "Call Quality Metrics":
                with st.spinner("Calculating call metrics..."):
                    metrics_results = metrics.analyze(call_data)
                
                # Display results
                st.subheader("Call Quality Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overtalk Percentage", f"{metrics_results['overtalk_percentage']:.2f}%")
                
                with col2:
                    st.metric("Silence Percentage", f"{metrics_results['silence_percentage']:.2f}%")
                
                # Visualize metrics
                metrics.visualize_metrics(call_data)
                
                # Call timeline visualization
                st.subheader("Call Timeline")
                metrics.visualize_timeline(call_data)
                
                # Conversation details
                with st.expander("View Conversation"):
                    # Create a DataFrame for better display
                    df = pd.DataFrame(call_data)
                    df['duration'] = df['etime'].astype(float) - df['stime'].astype(float)
                    st.dataframe(df[['speaker', 'text', 'stime', 'etime', 'duration']])
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show instructions when no file is uploaded
        st.info("Please upload a YAML or JSON file to begin analysis.")
        st.markdown("""
        ### Analysis Types:
        
        1. **Profanity Detection**: Identifies if agents or borrowers used profane language
        2. **Privacy & Compliance**: Detects if agents shared sensitive information without identity verification
        3. **Call Quality Metrics**: Calculates overtalk and silence percentages for the call
        
        ### Approaches:
        
        - **Pattern Matching (Regex)**: Uses regular expressions to detect patterns
        - **Machine Learning/LLM**: Uses AI models for detection
        """)

if __name__ == "__main__":
    main()