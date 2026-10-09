// ==============================================================================
// DYTR Studio Module 01: Problem Definition & Scientific Framing
// Smart Container Component (Orchestrator & API Integration)
// Based on CRISP-DM Phase 1 (Business Understanding) & UCD (ISO 9241-210)
// Zero Emojis Policy - Bilingual RTL/LTR - 4-Step Academic Wizard
// ==============================================================================

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { academicDictionary } from '../../utils/academicDictionary';
import { problemService } from '../../services/api/problemService';
import TermTooltip from '../common/TermTooltip';
import StepProgressBar from '../common/StepProgressBar';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import ProblemStatementStep from './presentational/ProblemStatementStep';
import DiagnosticQuizStep from './presentational/DiagnosticQuizStep';
import TaskTypeSelectorStep from './presentational/TaskTypeSelectorStep';
import HypothesisFramingStep from './presentational/HypothesisFramingStep';

const TOTAL_STEPS = 4;

export default function ProblemDefinitionContainer() {
  const {
    language,
    setActiveModule,
    problemState,
    updateProblemState,
    diagnoseProblemNecessity,
  } = useStudioStore();

  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  const [currentStep, setCurrentStep] = useState(1);
  const [statement, setStatement] = useState(problemState.problemStatement || '');
  const [answers, setAnswers] = useState(problemState.diagnosticAnswers || {});
  const [selectedTask, setSelectedTask] = useState(problemState.taskType || 'SENTENCE_CLASSIFICATION');
  const [hypothesis, setHypothesis] = useState(problemState.hypothesis || {});

  // API State
  const [isLoading, setIsLoading] = useState(true);
  const [apiError, setApiError] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [diagnosticQuestions, setDiagnosticQuestions] = useState([]);
  const [taskTypes, setTaskTypes] = useState([]);
  const [foundationModels, setFoundationModels] = useState([]);

  // Fetch all domain data dynamically from API Service layer
  useEffect(() => {
    let isSubscribed = true;

    async function loadProblemMetadata() {
      try {
        setIsLoading(true);
        setApiError(null);

        const [scenariosRes, questionsRes, tasksRes, modelsRes] = await Promise.all([
          problemService.getScenarios(),
          problemService.getDiagnosticQuestions(),
          problemService.getTaskTypes(),
          problemService.getFoundationModels(),
        ]);

        if (isSubscribed) {
          setScenarios(scenariosRes);
          setDiagnosticQuestions(questionsRes);
          setTaskTypes(tasksRes);
          setFoundationModels(modelsRes);
        }
      } catch (err) {
        if (isSubscribed) {
          console.error('Failed to load problem definitions from API:', err);
          setApiError(err.message || 'Error communicating with API.');
        }
      } finally {
        if (isSubscribed) {
          setIsLoading(false);
        }
      }
    }

    loadProblemMetadata();

    return () => {
      isSubscribed = false;
    };
  }, []);

  const stepLabels = useMemo(
    () =>
      isAr
        ? ['ما المشكلة التي تحلها؟', 'هل تحتاج ذكاء اصطناعيا؟', 'ما نوع المهمة؟', 'صياغة الفرضية']
        : ['Problem Statement', 'AI Necessity Diagnostic', 'Task Type Selection', 'Hypothesis Framing'],
    [isAr]
  );

  const handleApplyScenario = useCallback((scenario) => {
    const text = isAr ? scenario.text_ar : scenario.text_en;
    setStatement(text);
    const suggested = scenario.taskType || problemService.suggestTaskType(text);
    setSelectedTask(suggested);
    updateProblemState({
      problemStatement: text,
      taskType: suggested,
      hypothesis: { ...hypothesis, taskDescription: text },
    });
  }, [isAr, hypothesis, updateProblemState]);

  const handleAnswerChange = useCallback((questionId, value) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  }, []);

  const handleNext = () => {
    if (currentStep === 1) {
      updateProblemState({ problemStatement: statement });
      const suggested = problemService.suggestTaskType(statement);
      setSelectedTask(suggested);
    } else if (currentStep === 2) {
      updateProblemState({ diagnosticAnswers: answers });
      diagnoseProblemNecessity();
    } else if (currentStep === 3) {
      updateProblemState({ taskType: selectedTask });
    }
    if (currentStep < TOTAL_STEPS) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 1) setCurrentStep(currentStep - 1);
  };

  const handleSaveAndProceed = () => {
    updateProblemState({
      problemStatement: statement,
      diagnosticAnswers: answers,
      taskType: selectedTask,
      hypothesis: hypothesis,
      isSaved: true,
    });
    setActiveModule('dataprep');
  };

  if (isLoading) {
    return (
      <div className="container" style={{ maxWidth: '1000px', margin: '0 auto', padding: '4rem 1.5rem', textAlign: 'center' }}>
        <div style={{ fontSize: '1.1rem', color: 'var(--text-secondary)' }}>
          {isAr ? 'جاري تحميل إعدادات وتوصيف المسألة الأكاديمية عبر واجهة API...' : 'Loading academic problem framing parameters from API...'}
        </div>
      </div>
    );
  }

  return (
    <div className="container" style={{ maxWidth: '1000px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Module Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <span className="badge badge-teal">
            <TermTooltip termKey="problem_framing" customLabel={d.module_problem || (isAr ? 'تحديد المشكلة' : 'Problem Framing')} />
          </span>
          {apiError && (
            <span className="badge badge-warning" style={{ fontSize: '0.75rem' }}>
              {isAr ? 'وضع الاسترجاع الاحتياطي (Fallback Mode)' : 'Offline Fallback'}
            </span>
          )}
        </div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
          {isAr ? 'المرحلة 01: تحديد المشكلة وتأطير الفرضية العلمية' : 'Stage 01: Problem Definition & Scientific Framing'}
        </h1>
        <p style={{ fontSize: '0.92rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          {isAr
            ? 'قبل كتابة الكود أو جمع البيانات: حدد لماذا تبني هذا النموذج، هل تحتاج الذكاء الاصطناعي فعلا، وما هي معايير النجاح القابلة للاختبار.'
            : 'Formulate your objectives, verify the necessity of AI versus deterministic rules, and establish testable hypotheses.'}
        </p>
      </div>

      {/* 4-Step Progress Bar */}
      <StepProgressBar
        currentStep={currentStep}
        totalSteps={TOTAL_STEPS}
        stepLabels={stepLabels}
        onStepClick={(step) => setCurrentStep(step)}
      />

      {/* Main Content by Step */}
      <div style={{ marginTop: '1.5rem' }}>
        {currentStep === 1 && (
          <ProblemStatementStep
            isAr={isAr}
            statement={statement}
            scenarios={scenarios}
            onStatementChange={setStatement}
            onApplyScenario={handleApplyScenario}
          />
        )}
        {currentStep === 2 && (
          <DiagnosticQuizStep
            isAr={isAr}
            questions={diagnosticQuestions}
            answers={answers}
            diagnosticResult={problemState.diagnosticResult || 'ai_needed'}
            onAnswerChange={handleAnswerChange}
          />
        )}
        {currentStep === 3 && (
          <TaskTypeSelectorStep
            isAr={isAr}
            taskTypes={taskTypes}
            selectedTask={selectedTask}
            onSelectTask={setSelectedTask}
          />
        )}
        {currentStep === 4 && (
          <HypothesisFramingStep
            isAr={isAr}
            hypothesis={hypothesis}
            foundationModels={foundationModels}
            onHypothesisChange={setHypothesis}
          />
        )}
      </div>

      {/* Footer Navigation Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: '2.5rem',
          paddingTop: '1.5rem',
          borderTop: '1px solid var(--border-subtle)',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        {currentStep === 1 ? (
          <button
            onClick={() => setActiveModule('academy')}
            className="btn btn-secondary"
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            <span>{isAr ? 'العودة إلى الأكاديمية (00)' : 'Back to Academy (00)'}</span>
          </button>
        ) : (
          <button
            onClick={handlePrev}
            className="btn btn-secondary"
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            <span>{isAr ? 'الخطوة السابقة' : 'Previous Step'}</span>
          </button>
        )}

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {currentStep < TOTAL_STEPS ? (
            <button
              onClick={handleNext}
              disabled={currentStep === 1 && statement.trim().length < 5}
              className="btn btn-primary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <span>{isAr ? 'الخطوة التالية' : 'Next Step'}</span>
              <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          ) : (
            <button
              onClick={handleSaveAndProceed}
              className="btn btn-primary"
              style={{
                background: 'var(--accent-teal)',
                color: '#ffffff',
                fontWeight: 700,
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}
            >
              <span>{isAr ? 'حفظ الفرضية والانتقال إلى تجهيز البيانات (02)' : 'Save Hypothesis & Proceed to Data Prep (02)'}</span>
              <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
