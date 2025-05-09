import argparse
from emoji_gen.data_utils import get_emoji_list, prune_emoji_list, download_emojis as downloadEmojiList

def main():
    parser = argparse.ArgumentParser(description="Dev CLI for EmojiGen")
    parser.add_argument("task", 
                        choices=["scrape", "prune", "download", "prepare"],
                        help="Choose a dev task (scrape, prune, download, prepare)"
                        )
    args = parser.parse_args()
    try:
        if args.task == "scrape":
            run_scrape()
        elif args.task == "prune":
            run_prune()
        elif args.task == "download":
            run_download()
        elif args.task == "prepare":
            run_scrape()
            run_prune()
            run_download() 
            print("✅ Finished preparing emoji data")
        
        # else case handled by parser (above)

    except Exception as e:
        print(f"Stopped due to error: {e}")

if __name__ == "__main__":
    main() 



def run_scrape():
    print("Scraping emojis...\n")
    get_emoji_list()

def run_prune():
    print("Pruning emoji list...\n")
    prune_emoji_list()

def run_download():
    print("Downloading emoji images...\n")
    downloadEmojiList()