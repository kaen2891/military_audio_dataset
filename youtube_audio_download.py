import pandas as pd
import yt_dlp
import os

if __name__ == '__main__':
    # Read csv file
    df = pd.read_csv('mad_dataset_annotation.csv')
    
    df = df.loc[:,['Youtube title', 'Video_num', 'url']]
    df = df.drop_duplicates()

    # Get video id
    video_id = df['Video_num'].tolist()
    video_urls = df['url'].tolist()
    
    # Generate save folder
    output_path = './data/MAD_dataset/wav_files'
    os.makedirs(output_path, exist_ok=True)

    failed_downloads = []
    failed_video_ids = []
    successful_downloads = []

    for num, url in zip(df['Video_num'],df['url']):
        if int(num) < 10:
            num = '00'+str(num)
        elif int(num) < 99:
            num = '0'+str(num)
        
        ydl_opts = {
            'format': 'bestaudio/best',  # Select the best audio quality
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',  # Extract audio using FFmpeg
                'preferredcodec': 'wav',  # Save audio in wav format
                'preferredquality': '192',  # Set audio quality
            }],
            'outtmpl': output_path +'/'+ str(num),  # Output file template
            'quiet': True,  # Suppress output messages
            
        }       
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                print("Downloaded:", url, "to", output_path +'/'+ str(num)+'.wav')
                successful_downloads.append(url)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                failed_downloads.append(url)
                failed_video_ids.append(num)
                    

        # print("Failed downloads:", failed_downloads)
        # print("Successful downloads:", successful_downloads)