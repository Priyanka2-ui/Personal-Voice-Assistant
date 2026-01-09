import * as React from 'react';
import type { MessageFormatter, ReceivedChatMessage } from '@livekit/components-react';
import { cn } from '@/lib/utils';
import { useChatMessage } from './hooks/utils';

export interface ChatEntryProps extends React.HTMLAttributes<HTMLLIElement> {
  /** The chat massage object to display. */
  entry: ReceivedChatMessage;
  /** Hide sender name. Useful when displaying multiple consecutive chat messages from the same person. */
  hideName?: boolean;
  /** Hide message timestamp. */
  hideTimestamp?: boolean;
  /** An optional formatter for the message body. */
  messageFormatter?: MessageFormatter;
}

export const ChatEntry = ({
  entry,
  messageFormatter,
  hideName,
  hideTimestamp,
  className,
  ...props
}: ChatEntryProps) => {
  const { message, hasBeenEdited, time, locale, name } = useChatMessage(entry, messageFormatter);

  const isUser = entry.from?.isLocal ?? false;
  const messageOrigin = isUser ? 'remote' : 'local';

  return (
    <li
      data-lk-message-origin={messageOrigin}
      title={time.toLocaleTimeString(locale, { timeStyle: 'full' })}
      className={cn('group flex flex-col gap-2', className)}
      {...props}
    >
      {(!hideTimestamp || !hideName || hasBeenEdited) && (
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          {!hideName && (
            <div className="flex items-center space-x-2">
              <div
                className={cn(
                  'flex h-6 w-6 items-center justify-center rounded-full text-xs font-medium text-white',
                  isUser
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600'
                    : 'bg-gradient-to-r from-gray-600 to-gray-700'
                )}
              >
                {isUser ? 'U' : 'T'}
              </div>
              <span className="font-medium text-gray-700 dark:text-white">
                {isUser ? 'You' : name}
              </span>
            </div>
          )}

          {!hideTimestamp && (
            <span className="font-mono opacity-70">
              {hasBeenEdited && '*'}
              {time.toLocaleTimeString(locale, { timeStyle: 'short' })}
            </span>
          )}
        </div>
      )}

      <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
        <div
          className={cn(
            'max-w-[85%] min-w-0 rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm',
            isUser
              ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
              : 'border border-gray-200/50 bg-white text-gray-900 dark:border-gray-700/50 dark:bg-gray-800 dark:text-white'
          )}
        >
          <p className="whitespace-pre-wrap">{message}</p>
        </div>
      </div>
    </li>
  );
};
