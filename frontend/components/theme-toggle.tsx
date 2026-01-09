'use client';

import { useEffect } from 'react';
import { THEME_MEDIA_QUERY } from '@/lib/utils';

export function ApplyThemeScript() {
  useEffect(() => {
    const doc = document.documentElement;

    function applySystemTheme() {
      doc.classList.remove('dark', 'light');
      if (window.matchMedia(THEME_MEDIA_QUERY).matches) {
        doc.classList.add('dark');
      } else {
        doc.classList.add('light');
      }
    }

    // Apply system theme immediately
    applySystemTheme();

    // Listen for system theme changes
    const mediaQuery = window.matchMedia(THEME_MEDIA_QUERY);
    mediaQuery.addEventListener('change', applySystemTheme);

    // Cleanup listener on unmount
    return () => {
      mediaQuery.removeEventListener('change', applySystemTheme);
    };
  }, []);

  return null;
}
