#! /bin/sh
mkdir -p res && \
./staticres.py size confs/email-Enron.json && \
./staticres.py size confs/ca-AstroPH.json && \
./staticres.py size confs/cit-HepPh.json && \
./staticres.py size confs/p2p-Gnutella31.json && \
./staticres.py size confs/soc-Epinions1.json && \
./staticres.py size confs/com-dblp.json && \
./staticres.py size confs/loc-gowalla.json && \
./staticres.py size confs/as-skitter.json && \
./staticres.py size confs/com-youtube.json && \
./staticres.py time confs/time/email-Enron.json && \
./staticres.py time confs/time/ca-AstroPH.json && \
./staticres.py time confs/time/cit-HepPh.json && \
./staticres.py time confs/time/p2p-Gnutella31.json && \
./staticres.py time confs/time/soc-Epinions1.json && \
./staticres.py time confs/time/com-dblp.json && \
./staticres.py time confs/time/loc-gowalla.json && \
./progrres.py mult confs/progressive/email-Enron.json && \
./progrres.py mult confs/progressive/ca-AstroPH.json && \
./progrres.py mult confs/progressive/cit-HepPh.json && \
./progrres.py mult confs/progressive/p2p-Gnutella31.json && \
./progrres.py mult confs/progressive/soc-Epinions1.json && \
./progrres.py mult confs/progressive/com-dblp.json && \
./progrres.py mult confs/progressive/loc-gowalla.json && \
./progrres.py mult confs/progressive/as-skitter.json && \
./progrres.py mult confs/progressive/com-youtube.json
