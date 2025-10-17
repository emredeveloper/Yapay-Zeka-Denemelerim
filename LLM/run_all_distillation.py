"""
Main script to run all three distillation methods and compare results
"""

import sys
import time
import json
from datetime import datetime

def print_header(title):
    print("\n" + "=" * 100)
    print(f"{title:^100}")
    print("=" * 100 + "\n")

def main():
    print_header("LLM DISTILLATION TECHNIQUES COMPARISON")
    print("This script will run three different distillation methods:")
    print("1. Soft-Label Distillation - Full probability distribution transfer")
    print("2. Hard-Label Distillation - Memory-efficient, one-hot labels (DeepSeek style)")
    print("3. Co-Distillation - Simultaneous training of teacher and student (Llama 4 style)")
    print("\nEach method will be trained and evaluated separately.")
    
    all_results = {}
    
    # Method 1: Soft-Label Distillation
    try:
        print_header("METHOD 1: SOFT-LABEL DISTILLATION")
        print("Starting soft-label distillation training...")
        print("This uses the full probability distribution from the teacher model.")
        start_time = time.time()
        
        from soft_label_distillation import train_soft_label_distillation
        results_soft = train_soft_label_distillation()
        results_soft['training_time'] = time.time() - start_time
        all_results['soft_label'] = results_soft
        
        print(f"\n✓ Soft-label distillation completed in {results_soft['training_time']:.2f} seconds")
    except Exception as e:
        print(f"\n✗ Soft-label distillation failed: {str(e)}")
        all_results['soft_label'] = {"error": str(e)}
    
    # Method 2: Hard-Label Distillation
    try:
        print_header("METHOD 2: HARD-LABEL DISTILLATION")
        print("Starting hard-label distillation training...")
        print("This uses only the teacher's final predictions (memory efficient).")
        start_time = time.time()
        
        from hard_label_distillation import train_hard_label_distillation
        results_hard = train_hard_label_distillation()
        results_hard['training_time'] = time.time() - start_time
        all_results['hard_label'] = results_hard
        
        print(f"\n✓ Hard-label distillation completed in {results_hard['training_time']:.2f} seconds")
    except Exception as e:
        print(f"\n✗ Hard-label distillation failed: {str(e)}")
        all_results['hard_label'] = {"error": str(e)}
    
    # Method 3: Co-Distillation
    try:
        print_header("METHOD 3: CO-DISTILLATION")
        print("Starting co-distillation training...")
        print("This trains teacher and student simultaneously.")
        start_time = time.time()
        
        from co_distillation import train_co_distillation
        results_co = train_co_distillation()
        results_co['training_time'] = time.time() - start_time
        all_results['co_distillation'] = results_co
        
        print(f"\n✓ Co-distillation completed in {results_co['training_time']:.2f} seconds")
    except Exception as e:
        print(f"\n✗ Co-distillation failed: {str(e)}")
        all_results['co_distillation'] = {"error": str(e)}
    
    # Compare Results
    print_header("COMPARISON OF ALL METHODS")
    
    print("\n" + "=" * 100)
    print(f"{'Method':<30} {'Initial Loss':<15} {'Final Loss':<15} {'Improvement':<15} {'Time (s)':<15}")
    print("=" * 100)
    
    if 'soft_label' in all_results and 'error' not in all_results['soft_label']:
        r = all_results['soft_label']
        print(f"{'Soft-Label':<30} {r['initial_loss']:<15.4f} {r['final_loss']:<15.4f} "
              f"{r['improvement']:<15.4f} {r['training_time']:<15.2f}")
    
    if 'hard_label' in all_results and 'error' not in all_results['hard_label']:
        r = all_results['hard_label']
        print(f"{'Hard-Label (DeepSeek)':<30} {r['initial_loss']:<15.4f} {r['final_loss']:<15.4f} "
              f"{r['improvement']:<15.4f} {r['training_time']:<15.2f}")
    
    if 'co_distillation' in all_results and 'error' not in all_results['co_distillation']:
        r = all_results['co_distillation']
        print(f"{'Co-Distillation (Llama 4)':<30} {r['initial_student_loss']:<15.4f} "
              f"{r['final_student_loss']:<15.4f} {r['student_improvement']:<15.4f} "
              f"{r['training_time']:<15.2f}")
    
    print("=" * 100)
    
    # Key Insights
    print_header("KEY INSIGHTS")
    
    print("1. SOFT-LABEL DISTILLATION:")
    print("   ✓ Transfers full probability distribution")
    print("   ✓ Maximum knowledge transfer from teacher")
    print("   ✗ Requires significant memory (vocab_size × num_tokens)")
    print("   ✗ Needs access to teacher's weights")
    
    print("\n2. HARD-LABEL DISTILLATION (DeepSeek-R1):")
    print("   ✓ Very memory efficient (1 token ID vs vocab_size probabilities)")
    print("   ✓ Can work without teacher weights (just final outputs)")
    print("   ✓ Scalable to trillion-token datasets")
    print("   ✗ Less information transfer than soft labels")
    
    print("\n3. CO-DISTILLATION (Llama 4):")
    print("   ✓ Both models improve simultaneously")
    print("   ✓ Student benefits from evolving teacher")
    print("   ✓ No need for pre-trained teacher")
    print("   ✗ Requires training both models (more compute)")
    
    # Save results to JSON
    results_file = f"distillation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print_header("EXPERIMENT COMPLETED")
    print("All three distillation methods have been tested!")
    print("\nGenerated Models:")
    print("  • ./soft_label_student_model/")
    print("  • ./hard_label_student_model/")
    print("  • ./co_distillation_teacher_model/")
    print("  • ./co_distillation_student_model/")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
