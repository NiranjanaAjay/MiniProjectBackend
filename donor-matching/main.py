# ============================================================
# main.py
# Entry point — just the menu
# Command: python main.py
# ============================================================

from option_a import run_option_a
from option_b import run_option_b


if __name__ == '__main__':
    print("\n" + "="*55)
    print("     BONE MARROW TRANSPLANT MATCHING SYSTEM")
    print("="*55)
    
    while True:
        try:
            print("\n" + "─"*55)
            print("  MAIN MENU")
            print("─"*55)
            print("  1. Option A — Find top 5 donors for a patient")
            print("  2. Option B — Analyse a specific donor-patient pair")
            print("  3. Exit")
            print("─"*55)

            choice = input("  Enter choice (1/2/3): ").strip()

            if choice == '1':
                run_option_a()
            elif choice == '2':
                run_option_b()
            elif choice == '3':
                print("\n  Exiting. Goodbye! 👋\n")
                break
            else:
                print("  ⚠️  Enter 1, 2 or 3")

        except KeyboardInterrupt:
            print("\n\n  Exiting. Goodbye! 👋\n")
            break
