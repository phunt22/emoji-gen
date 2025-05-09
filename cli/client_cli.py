import argparse

# WORK IN PROGRESS
# BOILDER PLATE
def main():
  parser = argparse.ArgumentParser(description="EmojiGen CLI")
  parser.add_argument("--gen", help="Enter your emoji prompt!")
  args = parser.parse_args()
  print(f"Generating emoji for prompt: {args.gen}")

if __name__ == "__main__":
  main() 