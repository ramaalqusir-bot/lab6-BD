print("=" * 60)
print("Amdahl's Law")
print("=" * 60)

def S(P, N):
    return 1 / ((1-P) + P/N)

print("\nN\tP=0.5\tP=0.8\tP=0.9\tP=0.95\tP=0.99")
for N in [1,2,4,8,16,32,64,128,256]:
    print(f"{N}\t{S(0.5,N):.2f}\t{S(0.8,N):.2f}\t{S(0.9,N):.2f}\t{S(0.95,N):.2f}\t{S(0.99,N):.2f}")
