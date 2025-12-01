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

# 📑 Documento Executivo – Estrutura Geral do Repositório **MIHE/AGI**

Este documento consolida o **mapa visual da estrutura do repositório** `Agi_mire/`, servindo como referência executiva para auditoria, engenharia e certificação. Ele organiza os módulos, protocolos e serviços em uma visão clara e rastreável.

---

## 📂 Estrutura Geral antiga

```plaintext
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
│   ├── conftest.py
│   ├── thresholds.json
│   ├── logging.json
│   └── system.yaml
│
├── core/
│   ├── orchestration/
│   │     ├── mch.py
│   │     └── system_loop.py
│   │     
│   ├── intelligence/
│   │     ├── oa.py
│   │     ├── ol.py
│   │     ├── oea.py
│   │     └── ppo.py
│   │     
│   ├── governance/
│   │     ├── regvet.py
│   │     ├── simlog.py
│   │     └── prag.py
│   │     
│   ├── services/
│   │     ├── adaptation.py
│   │     ├── alert.py
│   │     ├── analytics.py
│   │     ├── attention.py
│   │     ├── control_bus.py
│   │     ├── monitor.py
│   │     ├── nlp_bridge.py
│   │     ├── pcvs.py
│   │     ├── perception.py
│   │     ├── security.py
│   │     ├── utils.py
│   │     ├── vector_index.py
│   │     └── multimodal/
│   │          ├── adapters/
│   │          │      ├── audio_bridge.py
│   │          │      └── vision_bridge.py
│   │          └── ovi_service/
│   │                 ├── ovi_core.py
│   │                 └── ovi_renderer.py 
│   │     
│   │     
│   ├── memory/     
│   │    └── hippocampus.py
│   │    
│   └── config_loader.py
│
├── demo/                     # Demonstrações completas
│   ├── rollback_demo.py
│   └── mch_cycle_demo.py
│
├── snapshots/
│   └── pcvs_0a8741294f54787b69abd2dd27f536a9bdc4699cd870f83ed0e4db972ba32ef5_1764045774937.json
│
├── tests/                    # Testes unitários/extensivos (pytest)
│
├── logs/                     # Logs persistentes
│   └── system.log
│
├── pcvs/
│   └── snapshots/
│          └── faiss_index_1764095192219.index
│
├── simulations/
│   └── auditoria_longitudinal.py
│
├── tools/
│   └── check_progress.py
│
├── main.py                   # Entrada principal do sistema
└── README.md                 # Este documento
```
---
## MAPA VISUAL DA ESTRUTURA HÍBRIDA (Agi_mire) Atual

## MAPA VISUAL DA ESTRUTURA HÍBRIDA (Agi_mire) - FINAL

Agi_mire/
│
├── data/                       # ARMAZENAMENTO FÍSICO E RASTREABILIDADE
│   ├── neuromorphic.ndb        # Append-Only Log (AOL)
│   ├── pcvs_snapshots/         # Snapshots de Rollback (Gerenciado pelo PRAG)
│   ├── persistent/             # Índices, KG e Catálogos
│   └── logs/                   # Logs de Sistema e Auditoria
│
├── configs/                    # PARÂMETROS GLOBAIS
│   ├── thresholds.json
│   └── system.yaml
│
└── core/                       # NÚCLEO CENTRAL DA AGI (Processamento)
    ├── storage/                # CAMADA FÍSICA E I/O DE BAIXO NÍVEL (DB Neuromórfico)
    │   ├── encoding/           # CAMADA 1: Codificação e Decodificação Binária
    │   ├── knowledge/          # Persistência do Pilar Simbólico (KG/Rule Base)
    │   │   ├── knowledge_graph_engine.py
    │   │   └── rule_base.py    
    │   ├── morphology/         # CAMADA 5: Plasticidade e Pesos
    │   │   └── plasticity_engine.py 
    │   ├── append_log_store.py # CAMADA 3: Consolidação (Escrita no AOL)
    │   ├── index_pointer.py    # Mapeamento VectorID -> Offset Físico
    │   └── bridge/             # CAMADA HÍBRIDA (Associação Lógica/Vetorial)
    │       ├── vector_index.py # CAMADA 2: Gerenciamento do FAISS/ANN (Morfologia)
    │       └── symbol_table.py 
    │
    ├── governance/             # GOVERNANÇA, ÉTICA E CONFORMIDADE
    │   ├── prag.py             # Trilha de Auditoria, Controlador de Rollback/PCVS
    │   ├── regvet.py
    │   └── simlog.py
    │
    ├── intelligence/           # AGENTES COGNITIVOS (Lógica)
    │   ├── oa.py               # Orquestrador Analítico
    │   ├── ol.py               # Orquestrador de Latência (Insight)
    │   ├── oea.py              # Orquestrador Ético
    │   └── ppo.py              # Meta-aprendizado
    │
    ├── orchestration/          # CICLO DE CONTROLE
    │   ├── mch.py              # Master Controller
    │   ├── system_loop.py
    │   └── control_bus.py # Mecanismo de Mensageria Interna
    │
    ├── memory/                 # API DA MEMÓRIA
    │   └── hippocampus.py      # get_topK(), write_memory(), rollback_call()
    │
    └── services/               # ADAPTADORES E SERVIÇOS DE SUPORTE
        ├── adaptation.py
        ├── attention.py
        ├── pcvs.py
        ├── utils.py
        ├── monitoring/
        │   ├── monitor.py
        │   └── alert.py # Emissão de Alertas
        │   └── analytics.py # Processamento de Logs e Análise de Longo Prazo
        ├── io/                 # Bridges de Entrada/Saída
        │   ├── nlp_bridge.py
        │   ├── perception.py # Coordenador de Entrada Sensorial de Alto Nível
        │   └── multimodal/
        └── security.py         # Módulo de Validação Ética (MVE)

## 🧩 Funções por Diretório

- **audit/** → Evidências de certificação, relatórios RST e auditorias longitudinais.  
- **checkpoints/** → Índices FAISS para busca vetorial persistente.  
- **configs/** → Configurações centrais (limiares, logging, thresholds).  
- **orchestration/** → Corpo Caloso (MCH) e loop principal do sistema.  
- **intelligence/** → Módulos cognitivos:  
  - `oa.py` (Razão),  
  - `ol.py` (Intuição),  
  - `oea.py` (Emoção/Ética),  
  - `ppo.py` (Evolução Estratégica).  
- **governance/** → Protocolos de sincronização e segurança: Sim-Log, Reg-Vet, PRAG.  
- **services/** → Serviços transversais (alertas, monitoramento, multimodalidade, segurança, indexação).  
- **memory/** → Hipocampo (memória contextual híbrida).  
- **demo/** → Scripts de demonstração (rollback, ciclo MCH).  
- **snapshots/** → Estados PCVS e registros de auditoria.  
- **tests/** → Testes unitários e integrados.  
- **logs/** → Logs persistentes do sistema.  
- **pcvs/** → Snapshots FAISS vinculados a pontos de verificação.  
- **simulations/** → Auditoria longitudinal simulada.  
- **tools/** → Utilitários de progresso e verificação.  
- **main.py** → Entrada principal.  
- **README.md** → Documentação raiz.  

---

## 📊 Valor Executivo

- **Rastreabilidade:** Estrutura garante logs, snapshots e evidências para auditoria externa.  
- **Modularidade:** Cada pasta reflete um módulo cognitivo ou protocolo de governança.  
- **Certificação:** Diretório `audit/` e `evidence/` são base para ISO/IEC 42001 e IEEE P7000.  
- **Explicabilidade:** `intelligence/` e `governance/` implementam protocolos híbridos (Sim-Log, Reg-Vet, PPO, PRAG).  
- **Resiliência:** `memory/` e `pcvs/` asseguram rollback e continuidade.  

---

📌 **Conclusão:** Este mapa visual é a espinha dorsal do relatório executivo. Ele demonstra que o MIHE/AGI possui **estrutura modular, governança ética, rastreabilidade e mecanismos de certificação**, consolidando sua prontidão para **auditoria externa e implementação em escala**.  


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


