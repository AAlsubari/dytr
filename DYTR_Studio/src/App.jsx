// ==============================================================================
// DYTR Studio Main Application - Full 12-Module Lifecycle Architecture
// Zero Emojis Policy - Bilingual RTL/LTR - Connected to dytr via FastAPI
// Modern Sidebar Navigation Architecture (RTL Right / LTR Left)
// Standards: ISO 9241-210 (UCD), CRISP-DM, DSR (Hevner et al., 2004)
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from './store/studioStore';
import SidebarNav from './components/navigation/SidebarNav';
import ActiveModelBar from './components/navigation/ActiveModelBar';
import PlatformAlertModal from './components/common/PlatformAlertModal';

// 12-Module Architecture Views
import AcademyWizardView from './components/academy/AcademyWizardView'; // 00
import ProblemDefinitionView from './components/problem/ProblemDefinitionView'; // 01
import DataMiningView from './components/dataprep/DataMiningView'; // 02
import ArchitectureStudioView from './components/builder/ArchitectureStudioView'; // 03
import VisualGlassroomView from './components/glassroom/VisualGlassroomView'; // 04
import TrainingArenaView from './components/arena/TrainingArenaView'; // 05
import TelemetryObservatoryView from './components/observatory/TelemetryObservatoryView'; // 06
import InferencePlaygroundView from './components/playground/InferencePlaygroundView'; // 07
import CodeExportView from './components/export/CodeExportView'; // 08
import EvaluationLabView from './components/evaluation/EvaluationLabView'; // 09
import DeploymentConsoleView from './components/deployment/DeploymentConsoleView'; // 10
import ContinualLearningView from './components/continual/ContinualLearningView'; // 11
import LandingShowcaseView from './components/landing/LandingShowcaseView'; // Showcase & Overview

export default function App() {
  const {
    activeModule,
    setActiveModule,
    language,
    theme,
    isSidebarCollapsed,
    setIsSidebarCollapsed,
    fetchCapabilities,
  } = useStudioStore();

  const handleToggleSidebar = (val) => {
    setIsSidebarCollapsed(val);
  };

  // 1. Initial Hardware & FastAPI Engine Capabilities Scan (Once on mount)
  useEffect(() => {
    fetchCapabilities();
  }, [fetchCapabilities]);

  // 2. Global Document Theme, Language & RTL/LTR Synchronization
  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.setAttribute('data-theme', theme);
  }, [language, theme]);

  // 3. Synchronize CSS variable for responsive layouts (e.g. sticky bottom preview)
  useEffect(() => {
    const sidebarWidth = activeModule === 'showcase' ? '0px' : (isSidebarCollapsed ? '68px' : '280px');
    document.documentElement.style.setProperty('--sidebar-width', sidebarWidth);
  }, [isSidebarCollapsed, activeModule]);

  // 4. Secret Developer / Administrator Keyboard Shortcut to access Landing & Showcase Mode (Ctrl+Shift+H or Alt+H)
  useEffect(() => {
    const handleSecretShortcut = (e) => {
      // Ctrl + Shift + H  OR  Alt + H  OR  Ctrl + Shift + L
      if (
        ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'H' || e.key === 'h' || e.key === 'L' || e.key === 'l')) ||
        (e.altKey && (e.key === 'h' || e.key === 'H' || e.key === 'l' || e.key === 'L'))
      ) {
        e.preventDefault();
        setActiveModule('showcase');
      }
    };
    window.addEventListener('keydown', handleSecretShortcut);
    return () => window.removeEventListener('keydown', handleSecretShortcut);
  }, [setActiveModule]);

  // 5. Lightweight Scroll Restoration on Stage Navigation
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    if (document.documentElement) document.documentElement.scrollTop = 0;
    if (document.body) document.body.scrollTop = 0;
  }, [activeModule]);

  // 6. Fast Bi-Directional (Bidi) Input Engine (Zero-Overhead Event Delegation)
  useEffect(() => {
    const isArabicText = (text) =>
      /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/.test(text);

    const handleInputDynamicBidi = (e) => {
      const target = e.target;
      if (!target || !['INPUT', 'TEXTAREA'].includes(target.tagName)) return;
      if (['url', 'email', 'number', 'password', 'date', 'checkbox', 'radio', 'range', 'file'].includes(target.type)) {
        target.setAttribute('dir', 'ltr');
        return;
      }
      if (target.classList?.contains('font-mono') || target.classList?.contains('code-input')) {
        target.setAttribute('dir', 'ltr');
        return;
      }

      const val = target.value || target.placeholder || '';
      const firstChar = val.trim().charAt(0);
      if (firstChar) {
        if (isArabicText(firstChar)) {
          target.setAttribute('dir', 'rtl');
        } else {
          target.setAttribute('dir', 'ltr');
        }
      } else {
        target.setAttribute('dir', 'auto');
      }
    };

    document.addEventListener('input', handleInputDynamicBidi, true);
    document.addEventListener('focus', handleInputDynamicBidi, true);

    return () => {
      document.removeEventListener('input', handleInputDynamicBidi, true);
      document.removeEventListener('focus', handleInputDynamicBidi, true);
    };
  }, []);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'row',
        background: 'var(--bg-primary)',
        overflowX: 'hidden',
      }}
    >
      {/* Interactive Academic Sidebar Navigation (Gemini Style: Fixed Viewport: RTL on Right / LTR on Left) */}
      {activeModule !== 'showcase' && (
        <SidebarNav
          isCollapsed={isSidebarCollapsed}
          setIsCollapsed={handleToggleSidebar}
        />
      )}

      {/* Main Studio Viewport Workspace */}
      <div
        style={{
          flex: 1,
          minWidth: 0,
          marginInlineStart: activeModule === 'showcase' ? 0 : (isSidebarCollapsed ? '68px' : '280px'),
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
          background: 'var(--bg-primary)',
          transition: 'margin-inline-start 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Global Active Model Bar & Platform Status (Visible in all stages except landing showcase) */}
        {activeModule !== 'showcase' && <ActiveModelBar />}

        {/* Main Stage Active View */}
        <main style={{ flex: 1, padding: activeModule === 'showcase' ? '0' : '0.5rem 0 2.5rem 0' }}>
          {activeModule === 'showcase' && <LandingShowcaseView />}
          {activeModule === 'academy' && <AcademyWizardView />}
          {activeModule === 'problem' && <ProblemDefinitionView />}
          {activeModule === 'dataprep' && <DataMiningView />}
          {activeModule === 'builder' && <ArchitectureStudioView />}
          {activeModule === 'arena' && <TrainingArenaView />}
          {activeModule === 'observatory' && <TelemetryObservatoryView />}
          {activeModule === 'evaluation' && <EvaluationLabView />}
          {activeModule === 'playground' && <InferencePlaygroundView />}
          {activeModule === 'export' && <CodeExportView />}
          {activeModule === 'continual' && <ContinualLearningView />}
          {activeModule === 'deployment' && <DeploymentConsoleView />}
          {activeModule === 'glassroom' && <VisualGlassroomView />}
        </main>

        {/* Academic Footer */}
        {activeModule !== 'showcase' && (
          <footer
            style={{
              marginTop: 'auto',
              padding: '1.25rem 2rem',
              borderTop: '1px solid var(--border-subtle)',
              background: 'var(--bg-secondary)',
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div>
                DYTR Studio · Academic Platform for Dynamic Transformers & Continual Learning
              </div>
              <div>
                Powered by <strong>dytr Core Library v0.1.3</strong> · Standard Architecture
              </div>
            </div>
          </footer>
        )}
      </div>

      {/* Global Studio Platform Alert Modal (Zero Browser Popups) */}
      <PlatformAlertModal />
    </div>
  );
}
