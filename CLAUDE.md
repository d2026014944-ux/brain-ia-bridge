# CLAUDE.md - Expansão: Painel de Controle (UI)

## Arquitetura da UI
- **Biblioteca**: PyGame (leve, baixo consumo de CPU, ideal para < 5W).
- **Desacoplamento**: O Cérebro roda em uma thread ou processo separado (ou via pooling assíncrono). A UI não dita o ritmo do disparo (Spiking).
- **Métricas de Visualização**:
  1. **Nível de Intenção**: Barra vertical (Input do HyperBitnet).
  2. **Grade de Neurônios**: Matriz de pixels (Brilham quando o `LIFNeuron` dispara e apagam suavemente devido ao fator "Leaky").
  3. **Coerência**: Círculo central que estabiliza via `quantum_states`.

## Regras de Interface (Neuromórfica)
A UI é apenas um espelho do cérebro. Ela não pode intervir no loop de simulação principal da **SpikingNetwork**.
As cores e a taxa de quadros (10 FPS no demonstrador básico) devem ser otimizadas para emular a percepção de pulso. Todo processamento de interface deve ser leve para economizar recursos e focar o desempenho no motor nativo `_native_core.cpp`.
