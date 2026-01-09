import type { AppConfig } from './lib/types';

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Priyanka Shilwant',
  pageTitle: 'Priyanka Shilwant - AI Voice Assistant',
  pageDescription: 'Experience a natural conversation with Priyanka, a GenAI Engineer Intern specializing in backend AI systems',

  supportsChatInput: true,
  supportsVideoInput: true,
  supportsScreenShare: true,
  isPreConnectBufferEnabled: true,

  logo: '/voice-icon.svg',
  accent: '#3b82f6',
  logoDark: '/voice-icon.svg',
  accentDark: '#60a5fa',
  startButtonText: 'Start Conversation',
};
