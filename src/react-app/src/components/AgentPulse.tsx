import React, { useEffect, useRef } from 'react';

interface LogEntry {
    time: number; // Nanoseconds or formatted string from backend
    message: string;
}

interface AgentPulseProps {
    logs: string[]; // Raw strings from backend "[Timestamp] Message"
}

export const AgentPulse: React.FC<AgentPulseProps> = ({ logs }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="agent-pulse-container" style={{
            background: '#0b0e11',
            border: '1px solid #2b3139',
            borderRadius: '4px',
            padding: '10px',
            height: '150px',
            overflowY: 'auto',
            fontFamily: 'monospace',
            fontSize: '12px',
            color: '#0ecb81'
        }}>
            <div style={{
                position: 'sticky',
                top: 0,
                background: '#0b0e11',
                borderBottom: '1px solid #2b3139',
                marginBottom: '5px',
                fontWeight: 'bold',
                color: '#848e9c'
            }}>
                ⚡ AGENT PULSE
            </div>
            <div ref={scrollRef}>
                {logs.length === 0 ? (
                    <div style={{ color: '#5e6673', fontStyle: 'italic' }}>Waiting for heartbeat...</div>
                ) : (
                    logs.map((log, i) => (
                        <div key={i} style={{ marginBottom: '2px' }}>
                            <span style={{ color: '#f0b90b', marginRight: '8px' }}>➜</span>
                            {log}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
