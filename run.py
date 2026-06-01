import argparse
import subprocess
import yaml
import sys

def main():
    parser = argparse.ArgumentParser(description="Run the Crash-Log Clusterer pipeline.")
    parser.add_argument('--skip-data', action='store_true', help="Skip data generation")
    parser.add_argument('--skip-embed', action='store_true', help="Skip embedding and clustering")
    parser.add_argument('--skip-label', action='store_true', help="Skip LLM labelling")
    parser.add_argument('--eval-only', action='store_true', help="Run only the evaluation script")
    
    args = parser.parse_args()
    
    # Ensure seed is set if used by scripts (though subprocesses load it from config.yaml directly)
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            seed = config.get("seed", 42)
    except FileNotFoundError:
        seed = 42

    def run_script(path):
        print(f"=====================================")
        print(f"🚀 Running {path}...")
        print(f"=====================================")
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"❌ Error running {path}. Exiting.")
            sys.exit(result.returncode)
            
    if args.eval_only:
        run_script("eval/run_eval.py")
        print("✅ Pipeline complete. Run: streamlit run src/dashboard.py")
        return

    if not args.skip_data:
        run_script("src/generate_data.py")
        
    if not args.skip_embed:
        run_script("src/embed_cluster.py")
        
    if not args.skip_label:
        run_script("src/llm_labeller.py")
        
    run_script("eval/run_eval.py")
    
    print("✅ Pipeline complete. Run: streamlit run src/dashboard.py")

if __name__ == "__main__":
    main()
