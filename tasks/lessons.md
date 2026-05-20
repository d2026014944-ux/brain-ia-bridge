# The Omega Protocol: Lessons Learned

*This file tracks root causes, dissected errors, and permanent rules to prevent repetition.*

## 🧠 Permanent Rules
1. **Feynman Filter**: Always explain the root mathematical cause before fixing.
2. **Anti-Vibe Coding**: Ensure every mathematical transformation (e.g., Fourier, phase locking) is covered by a test before merging into the core loop.

## 🐛 Dissected Errors
*(To be populated during Phase 4)*

## 🌊 Evolução Arquitetural
1. **Ondas Transformadas vs SNN Escalada:** Descobrimos que para simular o fenômeno de Aprendizado Subliminar (arXiv:2507.14805), pesos sinápticos fixos não funcionam. O peso deve ser uma onda (Amplitude x Cosseno com Fase) e a aprendizagem é o colapso/ressonância (phase-locking) em direção à fase do "Professor".
