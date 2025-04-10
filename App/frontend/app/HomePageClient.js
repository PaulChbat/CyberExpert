// File: app/HomePageClient.js

"use client";

import React, { useState, useEffect } from 'react'; // Added useEffect
import styles from './HomePage.module.css';
import Image from 'next/image';
import './Spinner.css'; // Import the CSS file



export default function HomePage() {
  // State hooks for main functionality
  const [logInput, setLogInput] = useState('');
  const [summary, setSummary] = useState(null); // Store intermediate summary
  const [actionResult, setActionResult] = useState(null); // Store final action
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- State hooks for Feedback ---
  const [suggestedActionInput, setSuggestedActionInput] = useState('');
  const [isFeedbackLoading, setIsFeedbackLoading] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState(null); // { type: 'success' | 'error', text: string }
  // ---

  // --- Backend API URLs ---
  const SUMMARY_BACKEND_URL = 'http://127.0.0.1:8000'; // Port for get_summary & feedback
  const ACTION_BACKEND_URL = 'http://127.0.0.1:8001'; // Port for get_action
  // ---

  // Reset feedback state whenever the main action result changes or clears
  useEffect(() => {
    setSuggestedActionInput('');
    setFeedbackMessage(null);
    setIsFeedbackLoading(false);
  }, [actionResult]); // Dependency array: runs when actionResult changes

  const handleGetActionClick = async () => {
    if (!logInput.trim()) {
      setError('Please enter some log data first.');
      setSummary(null);
      setActionResult(null);
      // Clear feedback state too
      setSuggestedActionInput('');
      setFeedbackMessage(null);
      return;
    }

    // Start loading, clear all previous results and feedback
    setIsLoading(true);
    setError(null);
    setSummary(null);
    setActionResult(null);
    setSuggestedActionInput('');
    setFeedbackMessage(null);
    setIsFeedbackLoading(false);

    let receivedSummary = null; // Variable to hold summary between calls

    // --- Step 1: Get Summary ---
    try {
      const encodedLog = encodeURIComponent(logInput);
      // Use SUMMARY_BACKEND_URL for the summary endpoint
      const summaryApiUrl = `${SUMMARY_BACKEND_URL}/get_summary/?log=${encodedLog}`;
      console.log("Fetching summary from:", summaryApiUrl);

      const summaryResponse = await fetch(summaryApiUrl);

      if (!summaryResponse.ok) {
        let errorDetail = `Summary Fetch Error: Status ${summaryResponse.status}`;
        try {
          const errorData = await summaryResponse.json();
          if (errorData.detail) {
            errorDetail = `Summary Fetch Error: ${errorData.detail}`;
          }
        } catch (jsonError) { /* Ignore if not JSON */ }
        throw new Error(errorDetail);
      }

      const summaryData = await summaryResponse.json();

      if (summaryData.summary) {
        receivedSummary = summaryData.summary;
        setSummary(receivedSummary);
        console.log("Received Summary:", receivedSummary);
      } else {
        throw new Error('Summary received, but content was missing.');
      }

    } catch (err) {
      if (err instanceof Error) {
        console.error("Summary API Call Failed:", err);
        setError(`Failed to fetch summary: ${err.message}. Ensure the summary backend (${SUMMARY_BACKEND_URL}) is running and CORS is configured correctly.`);
      } else {
        console.error("An unknown summary error occurred:", err);
        setError('An unexpected error occurred during summary fetch.');
      }
      setIsLoading(false);
      return;
    }

    // --- Step 2: Get Action (only if summary was successful) ---
    if (receivedSummary) {
      try {
        const encodedSummary = encodeURIComponent(receivedSummary);
        const actionApiUrl = `${ACTION_BACKEND_URL}/get_action/?log_summary=${encodedSummary}`;
        console.log("Fetching action from:", actionApiUrl);

        const actionResponse = await fetch(actionApiUrl);

        if (!actionResponse.ok) {
          let errorDetail = `Action Fetch Error: Status ${actionResponse.status}`;
          try {
            const errorData = await actionResponse.json();
            if (errorData.detail) {
              errorDetail = `Action Fetch Error: ${errorData.detail}`;
            }
          } catch (jsonError) { /* Ignore if not JSON */ }
          throw new Error(errorDetail);
        }

        const actionData = await actionResponse.json();

        if (actionData.action) {
          setActionResult(actionData.action);
          console.log("Received Action:", actionData.action);
          // Clear main error if action succeeds after summary error
          setError(null);
        } else {
          throw new Error('Action received, but content was missing.');
        }

      } catch (err) {
        if (err instanceof Error) {
          console.error("Action API Call Failed:", err);
          // Keep the summary visible, but show action error
          setError(`Successfully got summary, but failed to fetch action: ${err.message}. Ensure the action backend (${ACTION_BACKEND_URL}) is running and CORS is configured correctly.`);
        } else {
          console.error("An unknown action error occurred:", err);
          setError('An unexpected error occurred during action fetch.');
        }
        setActionResult(null); // Clear potential previous action
      } finally {
        setIsLoading(false);
      }
    } else {
       setIsLoading(false);
    }
  };

  // --- Feedback Submission Handler ---
  const handleFeedbackSubmit = async (event) => {
    event.preventDefault(); // Prevent default form submission if wrapped in <form>

    if (!suggestedActionInput.trim()) {
      setFeedbackMessage({ type: 'error', text: 'Please enter your suggested action.' });
      return;
    }
    if (!summary) {
      setFeedbackMessage({ type: 'error', text: 'Cannot submit feedback without a summary.' });
      return;
    }

    setIsFeedbackLoading(true);
    setFeedbackMessage(null);

    // Use SUMMARY_BACKEND_URL for the feedback endpoint
    const feedbackApiUrl = `${SUMMARY_BACKEND_URL}/feedback/?log_summary=${summary}&suggested_action=${suggestedActionInput}`; //`${ACTION_BACKEND_URL}/get_action/?log_summary=${encodedSummary}`;
    console.log("Submitting feedback to:", feedbackApiUrl);

    try {
      const response = await fetch(feedbackApiUrl);

      if (!response.ok) {
        let errorDetail = `Feedback Submit Error: Status ${response.status}`;
        try {
          const errorData = await response.json();
           // Adjust based on your actual error response structure
          errorDetail = `Feedback Submit Error: ${errorData.detail || response.statusText}`;
        } catch (jsonError) { /* Ignore if response is not JSON */ }
        throw new Error(errorDetail);
      }

      // Assuming the backend returns a success message on 2xx status
      const result = await response.json(); // Or handle based on backend response
      console.log("Feedback submission successful:", result);
      setFeedbackMessage({ type: 'success', text: 'Feedback submitted successfully! Thank you.' });
      setSuggestedActionInput(''); // Clear input on success

    } catch (err) {
      if (err instanceof Error) {
        console.error("Feedback API Call Failed:", err);
        setFeedbackMessage({ type: 'error', text: `Failed to submit feedback: ${err.message}. Ensure the backend (${feedbackApiUrl}) is running and CORS is configured.` });
      } else {
        console.error("An unknown feedback error occurred:", err);
        setFeedbackMessage({ type: 'error', text: 'An unexpected error occurred while submitting feedback.' });
      }
    } finally {
      setIsFeedbackLoading(false);
    }
  };
  // --- End Feedback ---


  // JSX structure updated
  return (
    <main className={styles.container}>
      <h1 className={styles.heading}>EXEO CyberExpert</h1>

      <textarea
        value={logInput}
        onChange={(e) => setLogInput(e.target.value)}
        placeholder="Paste your log data here..."
        rows={10}
        className={styles.textarea}
        disabled={isLoading || isFeedbackLoading} // Disable during both loading states
      />

      <button
        onClick={handleGetActionClick}
        className={`${styles.button} ${isLoading ? styles.buttonLoading : ''}`}
        disabled={isLoading || isFeedbackLoading} // Disable during both loading states
      >
        {isLoading ? 'Generating...' : 'Get Mitigation Action'}
      </button>

      {/* Display Area */}
      <div className={styles.resultsArea}>
        {/* Error Display */}
        {error && (
          <div className={styles.errorBox}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Summary Display (Only shown if Action hasn't arrived yet or Action failed) */}
        {summary && (
             <div className={styles.summaryBox} style={{marginBottom: '1rem'}}>
               <h2>Summary:</h2>
               <pre className={styles.summaryPre}>{summary}</pre>
             </div>
        )}

        {/* Action Display (Final Result) */}
        {actionResult && (
          <div className={styles.summaryBox}> {/* Reuse summaryBox style */}
            <h2>Mitigation Action:</h2>
            <pre className={styles.summaryPre}>
              {actionResult}
            </pre>
          </div>
        )}

        {/* Loading Indicator */}
        {isLoading && !error && (
          <div className="spinner-container">
            <div className="loading-spinner"></div>
          </div>
        )}

        {/* --- Feedback Section --- */}
        {/* Show only when an action is successfully displayed and not currently loading main action */}
        {actionResult && !isLoading && (
          <div className={styles.feedbackBox}>
            <h3>Provide Feedback on Action</h3>
            <p className={styles.feedbackInstructions}>
              If the suggested mitigation action isn't optimal, please provide a better one.
            </p>

            {/* Display the summary that led to this action */}
            <div className={styles.feedbackField}>
                <label className={styles.feedbackLabel}>Original Summary:</label>
                <pre className={`${styles.summaryPre} ${styles.feedbackSummaryDisplay}`}>
                    {summary}
                </pre>
            </div>


            {/* Input for suggested action */}
            <div className={styles.feedbackField}>
                <label htmlFor="suggestedAction" className={styles.feedbackLabel}>Your Suggested Action:</label>
                <textarea
                    id="suggestedAction"
                    value={suggestedActionInput}
                    onChange={(e) => setSuggestedActionInput(e.target.value)}
                    placeholder="Enter your suggested mitigation action here..."
                    rows={5}
                    className={styles.feedbackTextarea}
                    disabled={isFeedbackLoading}
                />
            </div>

            {/* Feedback Submit Button */}
            <button
                onClick={handleFeedbackSubmit}
                className={`${styles.button} ${styles.feedbackButton} ${isFeedbackLoading ? styles.buttonLoading : ''}`}
                disabled={isFeedbackLoading || !suggestedActionInput.trim()} // Disable if loading or input empty
            >
                {isFeedbackLoading ? 'Submitting...' : 'Submit Feedback'}
            </button>

            {/* Feedback Status Message */}
            {feedbackMessage && (
              <div className={`${styles.feedbackMessageBox} ${feedbackMessage.type === 'success' ? styles.feedbackSuccess : styles.feedbackError}`}>
                {feedbackMessage.text}
              </div>
            )}
          </div>
        )}
        {/* --- End Feedback Section --- */}

      </div>

      {/* Image Section */}
      <div style={{ marginTop: '2rem', textAlign: 'center' }}>
        <Image
          src="https://exeo.net/wp-content/uploads/2020/10/white-309x93.png" // Make sure this URL is correct
          width={165}
          height={50}
          alt="Exeo Logo"
          priority
        />
      </div>
    </main>
  );
}