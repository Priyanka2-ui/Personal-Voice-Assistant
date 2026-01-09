'use client';

import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import {
  type AgentState,
  type ReceivedChatMessage,
  useRoomContext,
  useVoiceAssistant,
} from '@livekit/components-react';
import { toastAlert } from '@/components/alert-toast';
import { ChatEntry } from '@/components/livekit/chat/chat-entry';
import { ChatMessageView } from '@/components/livekit/chat/chat-message-view';
import useChatAndTranscription from '@/hooks/useChatAndTranscription';
import { useDebugMode } from '@/hooks/useDebug';
import type { AppConfig } from '@/lib/types';
import { cn } from '@/lib/utils';
import { AudioVisualizer } from './waveform-bar';

function isAgentAvailable(agentState: AgentState) {
  return agentState == 'listening' || agentState == 'thinking' || agentState == 'speaking';
}

interface SessionViewProps {
  appConfig: AppConfig;
  disabled: boolean;
  sessionStarted: boolean;
}

export const SessionView = ({
  appConfig,
  disabled,
  sessionStarted,
  ref,
}: React.ComponentProps<'div'> & SessionViewProps) => {
  const { state: agentState } = useVoiceAssistant();
  const [chatOpen, setChatOpen] = useState(false);
  const { messages, send } = useChatAndTranscription();
  const room = useRoomContext();
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [displayMessages, setDisplayMessages] = useState<ReceivedChatMessage[]>([]);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);

  useDebugMode();

  // Clear chat when user starts speaking
  useEffect(() => {
    const hasUserMessage = messages.some((msg) => msg.from?.isLocal);
    if (hasUserMessage && !isUserSpeaking) {
      setIsUserSpeaking(true);
      // Clear previous messages when user starts talking
      setDisplayMessages([]);
    }
  }, [messages, isUserSpeaking]);

  // Update display messages when new messages come in
  useEffect(() => {
    if (messages.length > 0) {
      setDisplayMessages(messages.slice(-2)); // Show only latest Q&A pair
    }
  }, [messages]);

  // Auto-scroll to bottom when new messages arrive (for latest Q&A pair)
  useEffect(() => {
    if (chatContainerRef.current) {
      // Scroll to bottom to show latest messages
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [displayMessages]);

  // Also scroll to bottom on mount
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, []);

  // Function to ensure scroll to bottom
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      const scrollElement = chatContainerRef.current;
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  };

  // Scroll to bottom when component updates
  useEffect(() => {
    scrollToBottom();
  }, [displayMessages.length]);

  async function handleSendMessage(message: string) {
    // Clear chat history when user sends a message
    setDisplayMessages([]);
    setIsUserSpeaking(true);
    await send(message);
  }

  useEffect(() => {
    if (sessionStarted) {
      const timeout = setTimeout(() => {
        if (!isAgentAvailable(agentState)) {
          const reason =
            agentState === 'connecting'
              ? 'Agent did not join the room. '
              : 'Agent connected but did not complete initializing. ';

          toastAlert({
            title: 'Session ended',
            description: (
              <p className="w-full">
                {reason}
                Please try refreshing the page or check your connection.
              </p>
            ),
          });
          room.disconnect();
        }
      }, 10_000);

      return () => clearTimeout(timeout);
    }
  }, [agentState, sessionStarted, room]);

  // Determine if audio visualization should be active
  const isAudioActive = agentState === 'speaking' || isUserSpeaking;

  return (
    <div className="flex h-[100dvh] flex-col bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Top: Audio Visualizer */}
      <div className="flex items-center justify-center p-8 pt-16">
        <div className="text-center">
          <AudioVisualizer isActive={isAudioActive} className="mb-4" />
          <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            {agentState === 'speaking' && 'Agent is speaking...'}
            {agentState === 'listening' && 'Listening...'}
            {agentState === 'thinking' && 'Thinking...'}
            {agentState === 'connecting' && 'Connecting...'}
            {!isAgentAvailable(agentState) && 'Ready to chat'}
          </div>
        </div>
      </div>

      {/* Center: Chat area */}
      <main
        ref={ref}
        inert={disabled}
        className="mx-4 mb-4 flex flex-1 flex-col justify-center rounded-2xl bg-white/80 p-4 shadow-xl md:p-8 dark:bg-gray-900/80"
      >
        {/* Friendly prompt */}
        <div className="mt-2 mb-6 text-center">
          <p className="text-lg font-medium text-blue-700 dark:text-blue-200">
            Say &quot;hello&quot; to start the conversation!
          </p>
        </div>

        {/* Chat Messages - Centered */}
        <div className="flex flex-1 flex-col pt-4 pb-24">
          <div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto scroll-smooth px-4"
            style={{
              scrollBehavior: 'smooth',
              scrollbarWidth: 'thin',
              scrollbarColor: 'rgba(0,0,0,0.2) transparent',
            }}
          >
            <div className="mx-auto flex h-full max-w-4xl items-center justify-center">
              <div className="w-full max-w-3xl">
                <ChatMessageView className="h-auto">
                  <div className="space-y-4 whitespace-pre-wrap">
                    <AnimatePresence>
                      {displayMessages.length === 0 ? (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="py-12 text-center"
                        >
                          <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-blue-100 dark:bg-gray-800">
                            <svg
                              className="h-8 w-8 text-blue-500 dark:text-gray-400"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                              />
                            </svg>
                          </div>
                          <h3 className="mb-2 text-lg font-medium text-blue-700 dark:text-white">
                            Start the conversation
                          </h3>
                          <p className="mx-auto max-w-md text-blue-500 dark:text-white">
                            Ask me anything about my background, projects, or expertise in AI
                            development. I&apos;m here to help!
                          </p>
                        </motion.div>
                      ) : (
                        // Show only the latest Q&A pair
                        displayMessages.map((message: ReceivedChatMessage) => (
                          <motion.div
                            key={message.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, ease: 'easeOut' }}
                            className="rounded-2xl border border-gray-200 bg-white/90 p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900/90"
                          >
                            <ChatEntry hideName key={message.id} entry={message} />
                          </motion.div>
                        ))
                      )}
                    </AnimatePresence>
                  </div>
                </ChatMessageView>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
