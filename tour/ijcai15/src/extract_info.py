import sys
from fastkml import kml

def extract_info(fkml, fout):
    """Extract info from KML file"""
    items = []
    with open(fkml, 'r') as f:
        # read the content of KML file as a string
        kmlstr = f.read().replace('\n', '').replace('\t', '').replace("encoding='UTF-8'", '')

        # create a KML object
        ko = kml.KML()

        # read in the KML string
        ko.from_string(kmlstr)

        # the single <Document> element
        doc = list(ko.features())
        assert(len(doc) == 1)

        # <Placemark> elements of <Document>
        placemarks = list(doc[0].features())
        for p in placemarks:
            assert(isinstance(p, kml.Placemark))
            poi = p.name        # assigned POI name
            lng = p.geometry.x  # longitude
            lat = p.geometry.y  # latitude
            items.append((lng, lat, poi))

    with open(fout, 'w') as f:
        for item in items:
            line = str(item[0]) + ',' + str(item[1]) + ','
            if item[2]:
                line += str(item[2]).strip()
            f.write(line + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'KML_FILE  OUTPUT_FILE')
        sys.exit(0)

    fkml = sys.argv[1]
    fout = sys.argv[2]
    extract_info(fkml, fout)

