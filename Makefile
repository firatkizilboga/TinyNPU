ROOT := /home/firatkizilboga/compiler-optimization
PYTHON := python3
SYNTH_SCRIPT := $(ROOT)/scripts/synth_tinynpu_yosys.py
YOSYS_BIN := $(ROOT)/tools/install/yosys/bin/yosys
SLANG_PLUGIN := $(ROOT)/tools/src/yosys-slang/build/slang.so
WORKDIR ?= /tmp/tinynpu_synth_localrepo

.PHONY: help synth-full-ram synth-abstract-ram synth-blackbox-ub synth-blackbox-im stage-full-ram stage-abstract-ram stage-blackbox-ub stage-blackbox-im yosys-full-ram yosys-abstract-ram yosys-blackbox-ub yosys-blackbox-im

help:
	@echo "Targets:"
	@echo "  make synth-full-ram       # stage + run full-ram synthesis with live Yosys stdout"
	@echo "  make synth-abstract-ram   # stage + run abstract-ram synthesis with live Yosys stdout"
	@echo "  make synth-blackbox-ub    # stage + run with only unified_buffer blackboxed"
	@echo "  make synth-blackbox-im    # stage + run with only instruction_memory blackboxed"
	@echo "  make stage-full-ram       # only generate staged RTL and Yosys script"
	@echo "  make stage-abstract-ram   # only generate staged RTL and Yosys script"
	@echo "  make stage-blackbox-ub    # only generate staged RTL and Yosys script"
	@echo "  make stage-blackbox-im    # only generate staged RTL and Yosys script"
	@echo "Variables:"
	@echo "  WORKDIR=/tmp/tinynpu_synth_localrepo"
	@echo "  YOSYS_BIN=$(YOSYS_BIN)"
	@echo "  SLANG_PLUGIN=$(SLANG_PLUGIN)"

stage-full-ram:
	$(PYTHON) $(SYNTH_SCRIPT) --mode full-ram --workdir $(WORKDIR) --yosys-bin $(YOSYS_BIN) --slang-plugin $(SLANG_PLUGIN) --stage-only

stage-abstract-ram:
	$(PYTHON) $(SYNTH_SCRIPT) --mode abstract-ram --workdir $(WORKDIR) --yosys-bin $(YOSYS_BIN) --slang-plugin $(SLANG_PLUGIN) --stage-only

stage-blackbox-ub:
	$(PYTHON) $(SYNTH_SCRIPT) --mode blackbox-ub --workdir $(WORKDIR) --yosys-bin $(YOSYS_BIN) --slang-plugin $(SLANG_PLUGIN) --stage-only

stage-blackbox-im:
	$(PYTHON) $(SYNTH_SCRIPT) --mode blackbox-im --workdir $(WORKDIR) --yosys-bin $(YOSYS_BIN) --slang-plugin $(SLANG_PLUGIN) --stage-only

yosys-full-ram: stage-full-ram
	$(YOSYS_BIN) -Q -m $(SLANG_PLUGIN) $(WORKDIR)/synth_full_ram.ys

yosys-abstract-ram: stage-abstract-ram
	$(YOSYS_BIN) -Q -m $(SLANG_PLUGIN) $(WORKDIR)/synth_abstract_ram.ys

yosys-blackbox-ub: stage-blackbox-ub
	$(YOSYS_BIN) -Q -m $(SLANG_PLUGIN) $(WORKDIR)/synth_blackbox_ub.ys

yosys-blackbox-im: stage-blackbox-im
	$(YOSYS_BIN) -Q -m $(SLANG_PLUGIN) $(WORKDIR)/synth_blackbox_im.ys

synth-full-ram: yosys-full-ram

synth-abstract-ram: yosys-abstract-ram

synth-blackbox-ub: yosys-blackbox-ub

synth-blackbox-im: yosys-blackbox-im
