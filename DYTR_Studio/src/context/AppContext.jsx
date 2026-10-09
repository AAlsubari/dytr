// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Global App Context & State Manager

import React, { createContext, useContext, useState, useEffect } from 'react';
import { translations } from '../utils/translations';
import { api } from '../utils/api';
import useEngineStore from '../store/useEngineStore';

const AppContext = createContext();

export function AppProvider({ children }) {
  // Theme state: 'dark' (Default Slate Gray #1E2022) or 'light' (Off-White #F7F7F8)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('dytr_theme') || 'dark';
  });

  // Language state (default Arabic 'ar' with RTL)
  const [lang, setLang] = useState('ar');

  // Active view mode across 7 studios:
  // 'wizard', 'datasets', 'builder', 'arena', 'observatory', 'glassroom', 'playground'
  const [activeView, setActiveView] = useState('wizard');

  // Beginner Guide Modal
  const [isGuideOpen, setIsGuideOpen] = useState(false);

  // Synchronize theme with HTML data-theme attribute and persistence
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('dytr_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Wizard state
  const [selectedTask, setSelectedTask] = useState('task_sentiment');
  const [selectedBrain, setSelectedBrain] = useState('medium'); // 'small', 'medium', 'large'
  const [enableSIShield, setEnableSIShield] = useState(true);
  const [enableFlashAttn, setEnableFlashAttn] = useState(true);

  // Live Training & Model Readiness state
  const [trainingState, setTrainingState] = useState({
    isTraining: false,
    progress: 0,
    currentStepMessage: '',
    isTrained: false,
    loss: 1.5,
    accuracy: 85.0,
  });

  // System Hardware Capabilities
  const [capabilities, setCapabilities] = useState({
    cuda_available: false,
    device_name: 'Checking...',
    flash_attention_supported: true,
    vram_total_gb: 0,
  });

  // Update HTML dir and lang attributes when language changes
  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  }, [lang]);

  // Initialize communication layer engine store on load
  useEffect(() => {
    useEngineStore.getState().initialize().catch((err) => {
      console.warn('[AppContext] Engine store init error:', err);
    });
  }, []);

  // Fetch host capabilities on load
  useEffect(() => {
    api.getCapabilities().then((res) => {
      if (res) setCapabilities(res);
    });
  }, []);

  // Translation helper function
  const t = (key) => {
    return translations[lang]?.[key] || translations['en']?.[key] || key;
  };

  const toggleLanguage = () => {
    setLang((prev) => (prev === 'ar' ? 'en' : 'ar'));
  };

  return (
    <AppContext.Provider
      value={{
        theme,
        toggleTheme,
        lang,
        setLang,
        toggleLanguage,
        t,
        activeView,
        setActiveView,
        isGuideOpen,
        setIsGuideOpen,
        selectedTask,
        setSelectedTask,
        selectedBrain,
        setSelectedBrain,
        enableSIShield,
        setEnableSIShield,
        enableFlashAttn,
        setEnableFlashAttn,
        trainingState,
        setTrainingState,
        capabilities,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
