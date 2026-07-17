#! /bin/sh
mkdir -p logs  && \
./staticexp.py size confs/email-Enron.json && \
./staticexp.py size confs/ca-AstroPH.json && \
./staticexp.py size confs/cit-HepPh.json && \
./staticexp.py size confs/p2p-Gnutella31.json && \
./staticexp.py size confs/soc-Epinions1.json && \
./staticexp.py size confs/com-dblp.json && \
./staticexp.py size confs/loc-gowalla.json && \
./staticexp.py size confs/as-skitter.json && \
./staticexp.py size confs/com-youtube.json && \
./staticexp.py time confs/time/email-Enron.json && \
./staticexp.py time confs/time/ca-AstroPH.json && \
./staticexp.py time confs/time/cit-HepPh.json && \
./staticexp.py time confs/time/p2p-Gnutella31.json && \
./staticexp.py time confs/time/soc-Epinions1.json && \
./staticexp.py time confs/time/loc-gowalla.json && \
./staticexp.py time confs/time/com-dblp.json && \
./progrexp.py mult confs/progressive/email-Enron.json && \
./progrexp.py mult confs/progressive/ca-AstroPH.json && \
./progrexp.py mult confs/progressive/cit-HepPh.json && \
./progrexp.py mult confs/progressive/p2p-Gnutella31.json && \
./progrexp.py mult confs/progressive/soc-Epinions1.json && \
./progrexp.py mult confs/progressive/com-dblp.json && \
./progrexp.py mult confs/progressive/loc-gowalla.json && \
./progrexp.py mult confs/progressive/as-skitter.json && \
./progrexp.py mult confs/progressive/com-youtube.json
