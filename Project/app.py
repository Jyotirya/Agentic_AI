"""
Streamlit App - Interactive UI for IITB EE RAG Bot
This module provides a rich user interface with conversation history, sources, and feedback.
"""

import streamlit as st
from graph.supervisor import get_supervisor
from datetime import datetime
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="IITB EE RAG Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "supervisor" not in st.session_state:
    st.session_state.supervisor = get_supervisor()

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Sidebar
with st.sidebar:
    st.markdown("### 🎓 About")
    st.markdown("""
    This is an intelligent assistant for the IITB Electrical Engineering Department.
    
    **Features:**
    - Course information & prerequisites
    - Faculty research interests
    - Department announcements
    - Contextual answers with sources
    """)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("Messages", len(st.session_state.messages))
    
    # Collection info
    try:
        stats = st.session_state.supervisor.embedding_manager.get_collection_stats()
        st.markdown(f"**Documents:** {stats.get('total_documents', 0)}")
    except:
        st.markdown("**Documents:** Loading...")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    show_sources = st.checkbox("Show Sources", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)
    show_verification = st.checkbox("Show Verification Details", value=True)
    show_conflicts = st.checkbox("Show Conflict Warnings", value=True)
    show_metadata = st.checkbox("Show Metadata", value=False)
    
    # Query type filter
    query_filter = st.selectbox(
        "Filter by type",
        ["All", "Course", "Faculty", "Research", "Announcement"],
        key="query_filter"
    )
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.feedback = {}
        st.rerun()
    
    # Export history
    if st.button("💾 Export History", use_container_width=True):
        if st.session_state.messages:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "query_count": st.session_state.query_count
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Main content
st.markdown('<div class="main-header">📚 IITB Electrical Department Assistant</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
Ask me anything about courses, faculty, research, or department announcements!
</div>
""", unsafe_allow_html=True)

# Display conversation history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show additional info for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            # Confidence score
            if show_confidence and "confidence" in metadata:
                confidence = metadata["confidence"]
                if confidence is not None:
                    conf_percent = confidence * 100
                    color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"**Confidence:** <span style='color: {color}'>{conf_percent:.1f}%</span>", unsafe_allow_html=True)
            
            # Verification status
            if show_verification and "verification_status" in metadata:
                status = metadata.get("verification_status", "unknown")
                risk = metadata.get("risk_level", "unknown")
                
                status_icons = {
                    "verified": "✅",
                    "partially_verified": "⚠️",
                    "needs_review": "🔍",
                    "unverified": "❌",
                    "skipped": "⏭️"
                }
                risk_colors = {
                    "low": "green",
                    "moderate": "orange",
                    "high": "red",
                    "critical": "darkred"
                }
                
                st.markdown(
                    f"**Status:** {status_icons.get(status, '❓')} {status.replace('_', ' ').title()} | "
                    f"**Risk:** <span style='color: {risk_colors.get(risk, 'gray')}'>{risk.title()}</span>",
                    unsafe_allow_html=True
                )
            
            # Conflict warnings
            if show_conflicts and "conflict_report" in metadata:
                conflict_report = metadata.get("conflict_report", {})
                if conflict_report and not conflict_report.get("conflict_free", True):
                    conflicts = conflict_report.get("conflicts", [])
                    with st.expander(f"⚠️ {len(conflicts)} Conflict(s) Detected"):
                        for i, conflict in enumerate(conflicts[:3], 1):
                            st.warning(f"**{i}. {conflict.get('conflict_type', 'Unknown')}:** {conflict.get('description', 'No description')}")
            
            # Query type
            if show_metadata and "query_type" in metadata:
                st.markdown(f"**Query Type:** {metadata['query_type'].title()}")
            
            # Sources
            if show_sources and "context" in metadata:
                with st.expander("📄 View Sources"):
                    for i, ctx in enumerate(metadata["context"][:3]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f'<div class="source-box">{ctx[:300]}...</div>', unsafe_allow_html=True)
            
            # Feedback
            feedback_key = f"feedback_{idx}"
            if feedback_key not in st.session_state.feedback:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{idx}"):
                        st.session_state.feedback[feedback_key] = "positive"
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎", key=f"down_{idx}"):
                        st.session_state.feedback[feedback_key] = "negative"
                        st.info("Feedback noted. We'll improve!")
            else:
                feedback_type = st.session_state.feedback[feedback_key]
                st.markdown(f"*Feedback: {feedback_type}*")

# Chat input
if question := st.chat_input("Ask your question here..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke supervisor
                result = st.session_state.supervisor.invoke({"question": question})
                
                # Extract answer
                answer = result.get("answer", "I couldn't generate an answer.")
                
                # Display answer
                st.markdown(answer)
                
                # Store message with metadata
                message_data = {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "confidence": result.get("confidence"),
                        "query_type": result.get("query_type"),
                        "context": result.get("context", []),
                        "context_scores": result.get("context_scores", []),
                        "verification_status": result.get("verification_status"),
                        "risk_level": result.get("risk_level"),
                        "citations": result.get("citations"),
                        "conflict_report": result.get("conflict_report"),
                        "confidence_report": result.get("confidence_report")
                    }
                }
                st.session_state.messages.append(message_data)
                
                # Update query count
                st.session_state.query_count += 1
                
                # Show confidence
                if show_confidence and result.get("confidence") is not None:
                    confidence = result["confidence"]
                    conf_percent = confidence * 100
                    color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"**Confidence:** <span style='color: {color}'>{conf_percent:.1f}%</span>", unsafe_allow_html=True)
                
                # Show verification status
                if show_verification and result.get("verification_status"):
                    status = result.get("verification_status", "unknown")
                    risk = result.get("risk_level", "unknown")
                    
                    status_icons = {
                        "verified": "✅",
                        "partially_verified": "⚠️",
                        "needs_review": "🔍",
                        "unverified": "❌"
                    }
                    risk_colors = {
                        "low": "green",
                        "moderate": "orange",
                        "high": "red",
                        "critical": "darkred"
                    }
                    
                    st.markdown(
                        f"**Verification:** {status_icons.get(status, '❓')} {status.replace('_', ' ').title()} | "
                        f"**Risk:** <span style='color: {risk_colors.get(risk, 'gray')}'>{risk.title() if risk else 'Unknown'}</span>",
                        unsafe_allow_html=True
                    )
                
                # Show conflict warnings
                if show_conflicts and result.get("conflict_report"):
                    conflict_report = result.get("conflict_report", {})
                    if not conflict_report.get("conflict_free", True):
                        conflicts = conflict_report.get("conflicts", [])
                        with st.expander(f"⚠️ {len(conflicts)} Conflict(s) Detected"):
                            for i, conflict in enumerate(conflicts[:3], 1):
                                st.warning(f"**{i}. {conflict.get('conflict_type', 'Unknown')}:** {conflict.get('description', 'No description')}")
                            recommendations = conflict_report.get("recommendations", [])
                            if recommendations:
                                st.info("**Recommendations:**\n" + "\n".join(f"• {r}" for r in recommendations[:3]))
                
                # Show query type
                if show_metadata and result.get("query_type"):
                    st.markdown(f"**Query Type:** {result['query_type'].title()}")
                
                # Show sources with enhanced citation info
                if show_sources and (result.get("citations") or result.get("context")):
                    with st.expander("📄 View Sources & Citations"):
                        # Use citations if available, fallback to context
                        if result.get("citations"):
                            citations = result["citations"]
                            st.markdown(f"**{len(citations)} citation(s) from {len(set(c.get('source_file', '') for c in citations))} source(s)**")
                            
                            for i, citation in enumerate(citations[:5], 1):
                                source_file = citation.get("source_file", "unknown")
                                source_type = citation.get("source_type", "unknown")
                                relevance = citation.get("relevance_score", 0)
                                reliability = citation.get("reliability", "unknown")
                                snippet = citation.get("content_snippet", "")[:250]
                                
                                st.markdown(f"**[{i}] {source_type.title()} - {source_file}**")
                                st.markdown(f"*Relevance: {relevance:.0%} | Reliability: {reliability.title()}*")
                                st.markdown(f'<div class="source-box">{snippet}...</div>', unsafe_allow_html=True)
                        else:
                            context_list = result.get("context", [])
                            scores = result.get("context_scores", [])
                            
                            for i, ctx in enumerate(context_list[:3]):
                                st.markdown(f"**Source {i+1}:**")
                                if scores and i < len(scores):
                                    st.markdown(f"*Relevance: {scores[i]:.2f}*")
                                st.markdown(f'<div class="source-box">{ctx[:300]}...</div>', unsafe_allow_html=True)
                
                # Feedback buttons
                msg_idx = len(st.session_state.messages) - 1
                feedback_key = f"feedback_{msg_idx}"
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{msg_idx}"):
                        st.session_state.feedback[feedback_key] = "positive"
                        st.success("Thank you!")
                with col2:
                    if st.button("👎", key=f"down_{msg_idx}"):
                        st.session_state.feedback[feedback_key] = "negative"
                        st.info("Noted!")
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
    
    # Force rerun to update sidebar stats
    st.rerun()

# Example questions
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - What are the prerequisites for EE720?
        - Who teaches Advanced Power Electronics?
        - Tell me about Prof. R. K. Sharma's research
        """)
    
    with col2:
        st.markdown("""
        - What courses are available in power electronics?
        - Are there any upcoming seminars?
        - Which faculty work on heat and mass transfer?
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    IITB Electrical Engineering Department RAG Assistant | Built with Streamlit, LangGraph & ChromaDB
</div>
""", unsafe_allow_html=True)
