# Genre classifier

This is a musical genre classification tool that uses a convolutional neural network to predict the genre of a given song, provided by the user via a YouTube link. 

It is based on the brilliant work and tutorials of [Valerio Velardo](https://github.com/musikalkemist) and built with the [MARSAYAS](http://marsyas.info/downloads/datasets.html) dataset which consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

The project was built using Test Driven Development, and includes a suite of tests. The REST API was developed using Flask.

## Installation

Install the requirements.

```bash
pip install -r requirements.txt
```

## Usage
Run the server:
```python
python app.py
```
Navigate to localhost:5000 on your browser.

Enter a valid YouTube link to a musical track <15 minutes long.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
