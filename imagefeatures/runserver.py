__author__ = 'jason'

from rest import app
app.run(port=8099, host='0.0.0.0')
# app.run(port=app.config['PORT'], host='0.0.0.0')