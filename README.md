# 🧠 **MIHE-AGI — Arquitetura Híbrida Neuro-Simbolista**

### **Plataforma Cognitiva Certificada — Versão Interna V1.3**

Este repositório contém a **implementação oficial e auditável da Arquitetura MIHE/AGI**, composta por módulos neurossimbólicos, governança de segurança, ontogênese adaptativa e persistência determinística via snapshots PCVS.

A versão **V1.3** coloca foco em:

* **Determinismo sistêmico**
* **Governança e resiliência**
* **Rollback total e parcial**
* **Round-trip auditável**
* **Ontogênese (PPO) com triggers inteligentes**
* **FAISS estruturado com reconstrução total garantida**

---

# 📂 **Estrutura Geral do Repositório**

```
Agi_mire/
│
├── audit/                    # Auditoria e certificação
│   ├── RST_Certified_V1.3.json
│   ├── RST_Certified_V1.3.md
│   ├── audit_longitudinal.py
│   └── evidence/             # Snapshots, logs e hashes
│
├── checkpoints/              # Índices FAISS persistidos
│   └── faiss_index_xxx.index
│
├── configs/                  # Configurações centrais
│   ├── thresholds.json
│   ├── logging.json
│   └── system.yaml
│
├── core/
├── orchestration/
│   ├── mch.py
│   └── system_loop.py
├── intelligence/
│   ├── oa.py
│   ├── ol.py
│   └── ppo.py
├── governance/
│   ├── regvet.py
│   ├── simlog.py
│   └── prag.py
├── services/
│   ├── pcvs.py
│   ├── monitor.py
│   ├── analytics.py
│   ├── alert.py
│   ├── adaptation.py
│   ├── attention.py
│   ├── control_bus.py
│   ├── nlp_bridge.py
│   ├── perception.py
│   ├── security.py
│   ├── utils.py
│   └── multimodal/
│       ├── audio_bridge.py
│       └── vision_bridge.py
├── memory/
│   └── hippocampus.py
│
├── demo/                     # Demonstrações completas
│   ├── rollback_demo.py
│   └── mch_cycle_demo.py
│
├── snapshots/                # PCVS snapshots completos
│
├── tests/                    # Testes unitários/extensivos (pytest)
│
├── logs/                     # Logs persistentes
│
├── main.py                   # Entrada principal do sistema
└── README.md                 # Este documento
```

---

# 🧩 **Arquitetura MIHE/AGI — Visão Conceitual**

A arquitetura é inspirada em princípios neuroanatômicos reais, porém implementada de forma determinística, auditável e modular.

### **Diagrama resumido**

```
                        ┌───────────────────────────┐
                        │             MCH           │
                        │      (Corpo Caloso)       │
                        └─────────────┬─────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
 ┌─────────▼─────────┐      ┌─────────▼─────────┐      ┌─────────▼─────────┐
 │        OL         │      │         OA         │      │   Hippocampus     │
 │  (Intuição)       │      │  (Razão e Axiomas) │      │  (Memória LT)     │
 └─────────▲─────────┘      └─────────▲─────────┘      └─────────▲─────────┘
           │                          │                          │
 ┌─────────▼─────────┐      ┌─────────▼─────────┐      ┌─────────▼─────────┐
 │      RegVet        │      │      SimLog        │     │       PCVS        │
 │ Coerção Vetorial   │      │ Round-Trip / RT    │     │ Snapshots          │
 └─────────▲─────────┘      └─────────▲─────────┘      └─────────▲─────────┘
           │                          │                          │
 ┌─────────▼─────────┐      ┌─────────▼─────────┐      ┌─────────▼─────────┐
 │        PPO         │      │       PRAG        │      │    Auditoria      │
 │ (Ontogênese)       │      │ Governança        │      │  Rastros/Logs     │
 └────────────────────┘      └───────────────────┘      └───────────────────┘
```

---

# ⚙️ **Módulos Principais e Responsabilidades**

## 🧠 **MCH — Corpo Caloso (Orquestrador Geral)**

Coordena todo fluxo cognitivo:

* recebe entrada simbólica/vetorial
* aciona OA/OL conforme contexto
* registra round-trip no SimLog
* ativa rollback via PRAG
* dispara ontogênese pelo PPO
* coordena snapshots PCVS

É o **cérebro executivo**.

---

## 📚 **Hippocampus V4 — Memória de Longo Prazo**

* Armazena memórias com FAISS ou fallback in-memory
* Suporta decaimento, consolidação e reconstrução
* Integra totalmente com PCVS (save/restore)
* Determinismo garantido via hashing e ordering

---

## 🧩 **OA — Organismo Analítico**

* Grafo de triplas simbólicas
* Reconstrução simbólica → vetorial (via anchors)
* PRM (Preferential Rule Matching)
* Serialização auditável

---

## 🌙 **OL — Ontologia Local (Intuição)**

* Fornece caminhos alternativos quando OA falha
* Normalizações, heurísticas e fallback

---

## 🛡️ **PRAG — Governança/Segurança**

Responsável por:

* Detectar divergência D
* Avaliar coerência C
* Validar hash simbólico
* Registrar audit trail
* Decidir rollback total/parcial

É a **linha de defesa cognitiva**.

---

## 🔁 **PCVS — Ponto de Controle Determinístico**

* Snapshots completos do estado do sistema
* Hash SHA-256 para auditoria
* Rollback 100% determinístico

Base da certificação interna.

---

## 🎯 **PPO — Motor de Ontogênese**

* Gera novas conexões/axiomas
* Atua quando há:

  * oportunidade cognitiva
  * erro sistêmico
* Fornece mudanças estruturais controladas

---

## 🧭 **RegVet — Coerção Vetorial**

* Remove componentes indevidos do embedding
* Reforça direções corretas
* Seleciona regras por força/certidão/ângulo
* Gera recomendações metacognitivas

---

## 🔍 **SimLog — Round-Trip e Reconstrução**

* Mede fidelidade vetorial
* Round-trip determinístico
* Log estruturado (tensorlog)
* Digest SHA-256 para auditoria

---

# 🧪 **Testes (Pytest) — Cobertura Atual ~83%**

A suíte de testes cobre:

* cenários de sucesso e falha
* rollback total/parcial
* ontogênese por:

  * oportunidade
  * erro sistêmico
* round-trip
* reconstrução FAISS
* snapshots PCVS
* integridade de grafo simbólico
* governança PRAG

Rodar testes:

```bash
pytest --cov=core --cov-report=term-missing
```

---

# 🚀 Execução do Sistema

## Rodar o MCH (ciclo cognitivo completo)

```bash
python main.py
```

## Demos oficiais

```bash
python demo/mch_cycle_demo.py
python demo/rollback_demo.py
```

## Auditoria longitudinal

```bash
python audit/audit_longitudinal.py
```

---

# 📊 **Cenários Demonstrados**

### **Ciclo Normal**

* D baixo
* C alto
* aprendizado consolidado

### **Rollback Parcial**

* 0.70 < D ≤ 0.85

### **Rollback Total**

* D > 0.85

### **Ontogênese por Oportunidade**

* D < 0.20
* C > 0.80

### **Ontogênese por Erro Sistêmico**

* E > Tau

---

# 🔐 Certificação Interna V1.3

Este repositório segue o modelo de certificação:

* hashes determinísticos
* rastros de auditoria completos
* testes de robustez em profundidade
* snapshots verificáveis
* reconstrução fiel de FAISS
* governança para divergência

Arquivos oficiais ficam em `audit/`.

---

# 📘 Conclusão

Este repositório implementa uma **arquitetura AGI híbrida**, auditável e determinística, capaz de:

* aprender
* corrigir-se
* explicar-se
* evoluir estruturalmente
* retroceder com precisão
* registrar tudo para auditoria

É a versão mais estável da linhagem MIHE/AGI.


