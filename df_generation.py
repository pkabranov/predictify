import pandas as pd
import json
import spotipy as sp
from spotipy.oauth2 import SpotifyClientCredentials
'''
DataFrame generation script
Necessary fields:
    CLIENT_ID: Spotify client ID key
    CLIENT_SECRET: Spotify client secret key
    PLAYLIST_ID: Spotify ID for the playlist from which a dataframe is to be returned
    CSV_PATH: Path to .csv file with play counts

df_gen returns a Pandas DataFrame of the form:
  Column1 | Column2 | Column3 |    Column4   |  Column5 |  Column6 | Column7 |   Column8   |    Column9
----------|---------|---------|--------------|----------|----------|---------|-------------|---------------
   artist |  song   |   id    |  play count  |   key    | loudness |  tempo  | duration_ms | time_signature

feed_gen returns a Pandas DataFrame of the form: (this is the DataFrame we feed into the net)
   Column1   | Column2 |  Column3  |  Column4 |  Column5    |    Column6
-------------|---------|-----------|----------|-------------|---------------
  play count |   key   | loudness  |   tempo  | duration_ms | time_signature
'''
CLIENT_ID = '0ae72c9b0d4948d5ba0206c4c350d3fe'
CLIENT_SECRET = 'c1db0c06c2df46b7b9219250080adc8d'
PLAYLIST_ID = '4Kvvbir7qtwhF5xIeRILOl'
CSV_PATH = 'drive/My Drive/predictify/data_table_2.csv'

#Authorization
token = sp.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
cache_token = token.get_access_token()
spotify = sp.Spotify(cache_token)

#Get IDs in specified playlist
def get_playlist_ids(playlist_id=PLAYLIST_ID):
  offset = 0
  track_ids = []
  while offset < spotify.user_playlist_tracks('pkabranov', playlist_id, offset=0).get('total'):
    results = spotify.user_playlist_tracks('pkabranov', playlist_id, limit=1, offset=offset)
    for item in results['items']:
      if not item in track_ids:
        track_ids.append(item['track']['id'])
    offset += 1
  return track_ids

#Get usable features for each track in track_ids
def get_playlist_features(track_ids):
  batch, interval = 0, 100
  all_usable_track_features = []

  while batch < len(track_ids) - interval:
    all_usable_track_features.extend(spotify.audio_features(track_ids[batch:(batch + interval)]))
    batch += interval
    if batch > len(track_ids) - interval:
      all_usable_track_features.extend(spotify.audio_features(track_ids[batch:(batch + len(track_ids) % interval)]))
    print(len(all_usable_track_features))

  usable_keys = ['duration_ms', 'key', 'loudness', 'tempo', 'time_signature']
  usable_track_features = []

  for track in all_usable_track_features:
    if track:
      usable_track_features.append({key: value for (key, value) in track.items() if key in usable_keys})

  return usable_track_features

def get_artist_song_ids(csv_path=CSV_PATH):
  with open(csv_path, newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      #row_count = sum([1 for row in spamreader])
      all_rows_df = pd.DataFrame(columns = ['artist', 'song', 'id', 'play count'])
      for row in spamreader:
          url = urllib.parse.unquote(row[1] + "%20artist:" + row[0])
          search_result = spotify.search(url, type='track', limit=1)
          try:
            curr_row_df = pd.DataFrame({'artist': [search_result.get('tracks')['items'][0]['artists'][0].get('name')], #artist name
                                        'song': [search_result.get('tracks')['items'][0]['name']], #song name
                                        'id': [search_result.get('tracks')['items'][0]['id']], #song id
                                        'play count': row[2]})
            all_rows_df = all_rows_df.append(curr_row_df)
          except:
            continue
      return all_rows_df
#Create a dataframe with the specified features for all song IDs in the playlist passed in
def df_gen(playlist_id=PLAYLIST_ID):
    ids = get_playlist_ids(playlist_id)
    feat = get_playlist_features(ids)
    feat_df = pd.DataFrame.from_dict(feat, orient = "columns")
    #add play count to features dataframe
    df = pd.merge(get_artist_song_ids(), feat_df, on='id')
    df.drop(0, axis=1)
    return df
#Generate feed dataframe whcih contains explicit x_train and y_train data
def feed_gen(playlist_id=PLAYLIST_ID):
    tot = df_gen(playlist_id)
    feed = tot.drop([0,1,2])
    return feed
