import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Cross-check: does the SIM also stall when crippled like the sm_120 TE path?
import nvfp4_validate as M
print("== SIM all-FP4, NO SR, NO RHT (matches sm_120 TE constraint) ==")
M.run("sim-noSR-noRHT", 1500, hp_tail=0, sr=False, rht=False)
print("\n== SIM all-FP4, WITH SR+RHT (techniques the card can't run) ==")
M.run("sim-SR+RHT", 1500, hp_tail=0, sr=True, rht=True)
