import xml.etree.ElementTree
from xml.dom import minidom


# (folder, filename, width, height, depth, possibility, name, xmin, ymin, xmax, ymax)
def save_as_xml(outname, folder, filename, width, height, depth, objlst):
    with open(outname, "wb") as fd:
        doc = minidom.Document()
        doc.version = None
        root = doc.createElement("annotation")
        doc.appendChild(root)

        nfolder = doc.createElement("folder")
        tfolder = doc.createTextNode(folder)
        nfolder.appendChild(tfolder)
        root.appendChild(nfolder)

        nfilename = doc.createElement("filename")
        tfilename = doc.createTextNode(filename)
        nfilename.appendChild(tfilename)
        root.appendChild(nfilename)

        nsize = doc.createElement("size")
        nwidth = doc.createElement("width")
        nwidth.appendChild(doc.createTextNode(str(width)))
        nsize.appendChild(nwidth)
        nheight = doc.createElement("height")
        nheight.appendChild(doc.createTextNode(str(height)))
        nsize.appendChild(nheight)
        ndepth = doc.createElement("depth")
        ndepth.appendChild(doc.createTextNode(str(depth)))
        nsize.appendChild(ndepth)
        root.appendChild(nsize)

        nobject = doc.createElement("object")

        for possibility, name, xmin, ymin, xmax, ymax in objlst:
            npb = doc.createElement("possibility")
            npb.appendChild(doc.createTextNode(str(possibility)))
            nobject.appendChild(npb)
            nname = doc.createElement("name")
            nname.appendChild(doc.createTextNode(name))
            nobject.appendChild(nname)

            nbndbox = doc.createElement("bndbox")
            nxmin = doc.createElement("xmin")
            nxmin.appendChild(doc.createTextNode(str(xmin)))
            nbndbox.appendChild(nxmin)
            nymin = doc.createElement("ymin")
            nymin.appendChild(doc.createTextNode(str(ymin)))
            nbndbox.appendChild(nymin)
            nxmax = doc.createElement("xmax")
            nxmax.appendChild(doc.createTextNode(str(xmax)))
            nbndbox.appendChild(nxmax)
            nymax = doc.createElement("ymax")
            nymax.appendChild(doc.createTextNode(str(ymax)))
            nbndbox.appendChild(nymax)
            nobject.appendChild(nbndbox)

        root.appendChild(nobject)

        fd.write(doc.toprettyxml(encoding="utf-8"))

    pass

'''
def save_as_xml(outname, folder, filename, width, height, depth, possibility, name, xmin, ymin, xmax, ymax):
    with open(outname, "wb") as fd:
        doc = minidom.Document()
        doc.version = None
        root = doc.createElement("annotation")
        doc.appendChild(root)

        nfolder = doc.createElement("folder")
        tfolder = doc.createTextNode(folder)
        nfolder.appendChild(tfolder)
        root.appendChild(nfolder)

        nfilename = doc.createElement("filename")
        tfilename = doc.createTextNode(filename)
        nfilename.appendChild(tfilename)
        root.appendChild(nfilename)

        nsize = doc.createElement("size")
        nwidth = doc.createElement("width")
        nwidth.appendChild(doc.createTextNode(str(width)))
        nsize.appendChild(nwidth)
        nheight = doc.createElement("height")
        nheight.appendChild(doc.createTextNode(str(height)))
        nsize.appendChild(nheight)
        ndepth = doc.createElement("depth")
        ndepth.appendChild(doc.createTextNode(str(depth)))
        nsize.appendChild(ndepth)
        root.appendChild(nsize)

        nobject = doc.createElement("object")
        npb = doc.createElement("possibility")
        npb.appendChild(doc.createTextNode(str(possibility)))
        nobject.appendChild(npb)
        nname = doc.createElement("name")
        nname.appendChild(doc.createTextNode(name))
        nobject.appendChild(nname)

        nbndbox = doc.createElement("bndbox")
        nxmin = doc.createElement("xmin")
        nxmin.appendChild(doc.createTextNode(str(xmin)))
        nbndbox.appendChild(nxmin)
        nymin = doc.createElement("ymin")
        nymin.appendChild(doc.createTextNode(str(ymin)))
        nbndbox.appendChild(nymin)
        nxmax = doc.createElement("xmax")
        nxmax.appendChild(doc.createTextNode(str(xmax)))
        nbndbox.appendChild(nxmax)
        nymax = doc.createElement("ymax")
        nymax.appendChild(doc.createTextNode(str(ymax)))
        nbndbox.appendChild(nymax)
        nobject.appendChild(nbndbox)

        root.appendChild(nobject)

        fd.write(doc.toprettyxml(encoding="utf-8"))

        pass

    pass

'''

if __name__ == '__main__':
    #save_as_xml("001.xml", "000001.png", "00001", 1024, 1024, 3, 0.98, "classN", 641.4, 648.8, 667, 700.2)
    #save_as_xml("002.xml", "000002.png", "00001", 1024, 1024, 3, 0.98, "classN", 641.4, 648.8, 667, 700.2)
    objlst = [(0.98, "classN", 641.4, 648.8, 667, 700.2),
              (0.99, "classNP", 641.4, 648.8, 667, 700.2)]

    save_as_xml("003.xml", "../test", "00001", 1024, 1024, 3, objlst)
