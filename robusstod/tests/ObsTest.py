from godot.core import astro, tempo, util, station, constants, ipfwrap, autodif as ad
from godot.model import common, interface
from godot.model import obs
from godot import cosmos

import unittest
import os
import numpy as np


# Example of a light time correction custom class
class MyLightTimeCorrection(obs.LightTimeCorrection):
    def __init__(self, refCenter, refAxes):
        obs.LightTimeCorrection.__init__(self)
        # NOTE refAxes and refCenter should be the same of the owlt object and it should be used in the eval and deval method
        self.__center = refCenter
        self.__axes = refAxes

    def configure(self, axes, ref, point1, point2):
        pass

    def eval(self, epoch1, epoch2, r1, r2, dir):
        return 1.0

    def deval(self, epoch1, epoch2, r1, r2, vec):
        return 1.0


class NullLightTimeCorrection(obs.LightTimeCorrection):
    def __init__(self, refCenter, refAxes):
        obs.LightTimeCorrection.__init__(self)
        # NOTE refAxes and refCenter should be the same of the owlt object and it should be used in the eval and deval method
        self.__center = refCenter
        self.__axes = refAxes

    def configure(self, axes, ref, point1, point2):
        pass

    def eval(self, epoch1, epoch2, r1, r2, dir):
        return 0.0

    def deval(self, epoch1, epoch2, r1, r2, vec):
        return 0.0

# Example of a user-defined clock object


class MyClock(obs.Clock):
    def __init__(self):
        # NOTE maybe timeScale can be given in the init function and then used in the other method
        obs.Clock.__init__(self)

    def delta(self, timeScale, epoch):
        pass

    def referenceTimeScale(self):
        pass

    def differenceClockWithReference(self, epoch):
        pass

    def differenceReferenceWithCoordinateTime(self, timeScale, ts):
        pass

# Example of a user-defined clock object


class MyTDBTT(obs.IntTDBTT):
    def __init__(self):
        # NOTE maybe timeScale can be given in the init function and then used in the other method
        obs.IntTDBTT.__init__(self)

    # can be epoch/epoch, pos/epoch, pos, vel
    def ttMinusTdb(self, epoch, pos=None, vel=None):
        if vel is None and pos is None:
            return 1.0
        elif vel is None:
            return 2.0
        else:
            return 3.0


class MyScalarTimeEval(common.ScalarTimeEvaluable):
    def __init__(self) -> None:
        super().__init__()

    def eval(self, epoch):
        return 1.0


class MyFreq(interface.ScalarTimeIntegrable):
    def __init__(self, freq) -> None:
        super().__init__()
        self.freq = freq

    def eval(self, epoch):
        return self.freq

    def integrate(self, epoch, duration):
        return self.freq * duration


class MyProperTime(interface.ScalarTimeEvaluableDeriv):
    def __init__(self) -> None:
        super().__init__()

    def eval(self, epoch):
        return 1.0

    def deriv(self, epoch):
        return 1.0


class Test(unittest.TestCase):
    def test_class_lt(self):
        util.suppressLogger()

        # Define frames object
        file = os.path.join(os.path.dirname(__file__), "universe.yml")
        config = cosmos.util.load_yaml(file)
        uni = cosmos.Universe(config)

        # Create axes/point id and bodies
        sun = uni.frames.pointId("Sun")
        earth = uni.frames.pointId("Earth")
        moon = uni.frames.pointId("Moon")
        icrf = uni.frames.axesId("ICRF")
        ssb = uni.frames.pointId("SSB")
        sun_bd = uni.bodies.get("Sun")

        # Define lt object and additional input
        dir = obs.Direction.Backward
        gamma = MyScalarTimeEval()
        vec = np.ones(3)
        vec_ad = ad.Vector3(vec)
        ltc = MyLightTimeCorrection(ssb, icrf)
        nltc = NullLightTimeCorrection(ssb, icrf)
        rltc = obs.RelativisticLightTimeCorrection(
            uni.frames, ssb, icrf, [sun_bd], gamma=gamma)
        # NOTE necessary to call eval method at the moment (last 2 inputs not used)
        rltc.configure(icrf, ssb, ssb, ssb)
        epoch = tempo.Epoch()
        xepoch = tempo.XEpoch()

        # Create owlt obect w and w/o ltc
        owlt = obs.OneWayLightTime(uni.frames, ssb, icrf)
        owlt_ltc = obs.OneWayLightTime(uni.frames, ssb, icrf, ltc)
        owlt_rltc = obs.OneWayLightTime(uni.frames, ssb, icrf, rltc)
        owlt_nltc = obs.OneWayLightTime(uni.frames, ssb, icrf, nltc)
        # Direction is forward
        epoch_LT = epoch+owlt_rltc.eval(epoch, sun, earth, dir)
        xepoch_LT = xepoch+owlt_rltc.eval(xepoch, sun, earth, dir)
        r1 = uni.frames.vector3(ssb, sun, icrf, epoch)
        r2 = uni.frames.vector3(ssb, earth, icrf, epoch_LT)
        xr1 = uni.frames.vector3(ssb, sun, icrf, xepoch)
        xr2 = uni.frames.vector3(ssb, earth, icrf, xepoch_LT)
        rltc.eval(epoch, epoch_LT, r1, r2, dir)
        rltc.eval(xepoch, xepoch_LT, xr1, xr2, dir).value()
        rltc.deval(epoch, epoch_LT, r1, r2, vec)
        rltc.deval(xepoch, xepoch_LT, xr1, xr2, vec_ad).value()

        # Test eval
        self.assertNotEqual(owlt_ltc.eval(epoch, sun, earth, dir), owlt.eval(
            epoch, sun, earth, dir))
        self.assertNotEqual(owlt_ltc.eval(xepoch, sun, earth, dir).value(
        ), owlt.eval(xepoch, sun, earth, dir).value())
        self.assertNotEqual(owlt_rltc.eval(epoch, sun, earth, dir), owlt.eval(
            epoch, sun, earth, dir))
        self.assertNotEqual(owlt_rltc.eval(xepoch, sun, earth, dir).value(
        ), owlt.eval(xepoch, sun, earth, dir).value())
        self.assertEqual(owlt_nltc.eval(epoch, sun, earth, dir), owlt.eval(
            epoch, sun, earth, dir))
        self.assertEqual(owlt_nltc.eval(xepoch, sun, earth, dir).value(
        ), owlt.eval(xepoch, sun, earth, dir).value())

        # Test deval
        self.assertNotEqual(owlt_ltc.deval(epoch, sun, earth, vec), owlt.deval(
            epoch, sun, earth, vec))
        self.assertNotEqual(owlt_ltc.deval(xepoch, sun, earth, vec_ad).value(
        ), owlt.deval(xepoch, sun, earth, vec_ad).value())
        self.assertNotEqual(owlt_rltc.deval(epoch, sun, earth, vec), owlt.deval(
            epoch, sun, earth, vec))
        self.assertNotEqual(owlt_rltc.deval(xepoch, sun, earth, vec_ad).value(
        ), owlt.deval(xepoch, sun, earth, vec_ad).value())
        self.assertEqual(owlt_nltc.deval(epoch, sun, earth, vec), owlt.deval(
            epoch, sun, earth, vec))
        self.assertEqual(owlt_nltc.deval(xepoch, sun, earth, vec_ad).value(
        ), owlt.deval(xepoch, sun, earth, vec_ad).value())

    def test_class_gs(self):
        util.suppressLogger()

        # Define frames object
        file = os.path.join(os.path.dirname(__file__), "universe.yml")
        config = cosmos.util.load_yaml(file)
        uni = cosmos.Universe(config)
        station_file = "share/test/GroundStationsDatabase.json"
        station_config = cosmos.util.load_json(station_file)
        station_book = station.StationBook(station_config, uni.constants)
        nn11 = station_book.get("NN11")

        # Define some inputs
        epoch = tempo.Epoch()
        xepoch = tempo.XEpoch()
        moon = uni.frames.pointId("Moon")

        # Media Corrections
        # NOTE the mappign coefficients list MUST have len == 10
        solmag = ipfwrap.IpfReader("share/test/solmag.ipf", 8)
        coeffs = list(np.ones(10))
        zWetPD = MyScalarTimeEval()
        zDryPD = MyScalarTimeEval()
        wetScale = MyScalarTimeEval()
        dryScale = MyScalarTimeEval()
        ionoScale = MyScalarTimeEval()
        mediaCorr = obs.MediaCorrections(
            uni.frames, "New_Norcia", "New_Norcia", "Sun", "Earth", "ITRF", "ICRF", solmag, coeffs)
        mediaCorr.setWetZenithDelay(zWetPD)
        mediaCorr.setDryZenithDelay(zDryPD)
        mediaCorr.setCorrections(wetScale, dryScale, ionoScale)
        mediaCorr.setContributions(False, False)
        self.assertEqual(mediaCorr.ltcorr(
            epoch, epoch+10, moon, 8.5e09, True, False), 0.0)
        self.assertEqual(
            mediaCorr.getLTCorrectionInfo().dryTropo, obs.Media.Type.ZERO)
        self.assertEqual(
            mediaCorr.getLTCorrectionInfo().wetTropo, obs.Media.Type.ZERO)
        self.assertEqual(
            mediaCorr.getLTCorrectionInfo().iono, obs.Media.Type.ZERO)
        mediaCorr.setContributions(True, True)
        self.assertNotEqual(mediaCorr.ltcorr(
            epoch, epoch+10, moon, 8.5e09, True, False), 0.0)
        self.assertEqual(
            mediaCorr.getLTCorrectionInfo().dryTropo, obs.Media.Type.METEO)
        self.assertEqual(
            mediaCorr.getLTCorrectionInfo().wetTropo, obs.Media.Type.METEO)
        self.assertEqual(
            mediaCorr.getLTCorrectionInfo().iono, obs.Media.Type.MEAN)

        # Clock
        myClock = MyClock()
        tdbtt = obs.TDBTT(eph_file="share/test/de432.jpl")
        myTDBTT = MyTDBTT()
        myStnClock = MyScalarTimeEval()
        gsClock = obs.GroundStationClock(
            uni.frames, "New_Norcia", "Earth", "ICRF", myTDBTT, myStnClock)
        self.assertEqual(gsClock.differenceClockWithReference(
            epoch), myStnClock.eval(epoch))
        # NOTE gsClock.differenceReferenceWithCoordinateTime pass as input to TDBTT.ttMinusTdb epoch and pos always
        self.assertEqual(gsClock.differenceReferenceWithCoordinateTime(tempo.TimeScale.TDB, epoch) -
                         gsClock.differenceReferenceWithCoordinateTime(tempo.TimeScale.TT, epoch), myTDBTT.ttMinusTdb(epoch, np.ones(3)))

        # GroundStationParticipant
        myFreq = common.ConstantScalarTimeEvaluable(1.0)
        myBias = MyScalarTimeEval()
        gsPar = obs.GroundStationParticipant(
            "stnParNN", "New_Norcia", "New_Norcia", nn11, gsClock, mediaCorr)
        gsPar.setRangeBiases({obs.FrequencyBand.X: myBias}, {
                             obs.FrequencyBand.X: myBias})
        self.assertEqual(gsPar.uplinkRangeBias(
            obs.FrequencyBand.X, epoch), 1.0)
        self.assertEqual(gsPar.uplinkRangeBias(
            obs.FrequencyBand.X, xepoch), ad.Scalar(1.0))
        self.assertEqual(gsPar.uplinkRangeBias(
            obs.FrequencyBand.Ka, epoch), 0.0)
        self.assertEqual(gsPar.uplinkRangeBias(
            obs.FrequencyBand.Ka, xepoch), ad.Scalar(0.0))
        gsPar.setTransmitterFrequency(myFreq)
        self.assertEqual(gsPar.getTransmitterFrequency(), myFreq)

    def test_class_sc(self):
        util.suppressLogger()

        # Define frames object
        file = os.path.join(os.path.dirname(__file__), "universe.yml")
        config = cosmos.util.load_yaml(file)
        uni = cosmos.Universe(config)

        # Create axes/point id and bodies
        sun_bd = uni.bodies.get("Sun")

        # Define some inputs
        epoch = tempo.Epoch()
        xepoch = tempo.XEpoch()
        groupDelay = MyScalarTimeEval()
        phaseDelay = MyScalarTimeEval()
        freq = common.ConstantScalarTimeEvaluable(1.0)
        tau = MyProperTime()
        bandComb = obs.BandCombination(
            freqUp=obs.FrequencyBand.X, freqDown=obs.FrequencyBand.Ka)
        bandComb_str = obs.BandCombination.fromString("X_X")
        trData = obs.TransponderData(
            groupDelay=groupDelay, phaseDelay=phaseDelay)
        transponder = obs.Transponder(config={"X_Ka": trData})

        # ProperTimeEquation
        prTimeEq = obs.ProperTimeEquation(
            uni.frames, "Moon", "SSB", "ICRF", 0.0, [sun_bd])

        # Band combination
        self.assertEqual(bandComb.up(), bandComb_str.up())
        self.assertNotEqual(bandComb.down(), bandComb_str.down())

        # Transponder
        transponder.add("S_X", trData)
        self.assertEqual(transponder.transponderRatio(bandComb), 3344/749)
        self.assertEqual(transponder.groupDelay(bandComb, xepoch).value(), 1.0)
        self.assertEqual(transponder.phaseDelay(bandComb, epoch), 1.0)

        # Spacecraft Participant
        scPar = obs.SpacecraftParticipant("Spacecraft", {"sc_antenna": "antenna#1"}, {
                                          "sc_transponder": {"X_Ka": trData}}, freq)
        self.assertEqual(scPar.getTransmitterFrequency(), freq)
        scPar.setProperTimeEquation(tau)
        tau1 = scPar.getProperTimeEquation()
        self.assertEqual(tau1.deriv(epoch), 1.0)
        self.assertEqual(scPar.transponders()[
                         "sc_transponder"].transponderRatio(bandComb), 3344/749)

    def test_class_obs(self):
        util.suppressLogger()

        # Define frames object
        file = os.path.join(os.path.dirname(__file__), "universe.yml")
        config = cosmos.util.load_yaml(file)
        uni = cosmos.Universe(config)
        uni.parameters.add("test_param", 1.0)
        uni.parameters.add("test_param1", 2.0)
        test_param = uni.parameters.get("test_param")
        test_param1 = uni.parameters.get("test_param1")
        uni.parameters.track(["test_param", "test_param1"])

        # Define some inputs
        epoch = tempo.Epoch()
        xepoch = tempo.XEpoch()
        scPar = obs.SpacecraftParticipant("Moon")
        icrf = uni.frames.axesId("ICRF")
        ssb = uni.frames.pointId("SSB")
        sun_bd = uni.bodies.get("Sun")

        # Position
        pos = obs.Position(uni.frames, scPar, "ICRF")
        self.assertEqual(list(pos.eval(epoch)), list(
            uni.frames.vector3("Earth", "Moon", "ICRF", epoch)))
        self.assertEqual(list(pos.eval(xepoch).value()), list(
            uni.frames.vector3("Earth", "Moon", "ICRF", xepoch).value()))

        # Direction w/o light time
        optDir = obs.OptDirection(uni.frames)
        optDir.setObserverPoint("Earth")
        optDir.setObserverAxes("ICRF")
        optDir.setTarget("Moon")
        self.assertEqual(list(optDir.eval(epoch)), list(
            uni.frames.vector3("Earth", "Moon", "ICRF", epoch)))
        self.assertEqual(list(optDir.eval(xepoch).value()), list(
            uni.frames.vector3("Earth", "Moon", "ICRF", xepoch).value()))

        # Direction w light time
        owlt = obs.OneWayLightTime(uni.frames, ssb, icrf)
        optDir_ref = obs.OptDirection(uni.frames, owlt, False)
        optDir_ref.setObserverPoint("Earth")
        optDir_ref.setObserverAxes("ICRF")
        optDir_ref.setTarget("Moon")
        optDir_ref_w_ab = obs.OptDirection(uni.frames, owlt, True)
        optDir_ref_w_ab.setObserverPoint("Earth")
        optDir_ref_w_ab.setObserverAxes("ICRF")
        optDir_ref_w_ab.setTarget("Moon")
        # check that aberration computation is done
        self.assertNotEqual(
            list(optDir_ref_w_ab.eval(epoch)), list(optDir_ref.eval(epoch)))
        self.assertNotEqual(list(optDir_ref_w_ab.eval(xepoch).value()), list(
            optDir_ref.eval(xepoch).value()))
        # check the possible light time correction
        nltc = NullLightTimeCorrection(ssb, icrf)
        ltc = MyLightTimeCorrection(ssb, icrf)
        gamma = MyScalarTimeEval()
        rltc = obs.RelativisticLightTimeCorrection(
            uni.frames, ssb, icrf, [sun_bd], gamma=gamma)
        lt_corrections = [nltc, ltc, rltc]
        for abFlag in [True, False]:
            for n_case, ltcorr in enumerate(lt_corrections):
                if abFlag:
                    optDir_comp = optDir_ref_w_ab
                else:
                    optDir_comp = optDir_ref
                owlt_corr = obs.OneWayLightTime(uni.frames, ssb, icrf, ltcorr)
                optDir = obs.OptDirection(uni.frames, owlt_corr, abFlag)
                optDir.setObserverPoint("Earth")
                optDir.setObserverAxes("ICRF")
                optDir.setTarget("Moon")
                if n_case == 0:
                    self.assertEqual(
                        list(optDir.eval(epoch)), list(optDir_comp.eval(epoch)))
                    self.assertEqual(list(optDir.eval(xepoch).value()), list(
                        optDir_comp.eval(xepoch).value()))
                else:
                    self.assertNotEqual(
                        list(optDir.eval(epoch)), list(optDir_comp.eval(epoch)))
                    self.assertNotEqual(list(optDir.eval(xepoch).value()), list(
                        optDir_comp.eval(xepoch).value()))

        # Radio observables
        gamma = common.ConstantScalarTimeEvaluable(1.0)
        rltc = obs.RelativisticLightTimeCorrection(
            uni.frames, ssb, icrf, [sun_bd], gamma=gamma)
        owlt = obs.OneWayLightTime(uni.frames, ssb, icrf, rltc)
        # Can be used both frequency formulation
        freq = common.ConstantScalarTimeEvaluable(8e9)
        freq1 = MyFreq(8e9)
        upBias = MyScalarTimeEval()
        dnBias = common.ConstantScalarTimeEvaluable(0.0)
        upDelay = common.ConstantScalarTimeEvaluable(0.0)
        dnDelay = MyScalarTimeEval()
        # Ground Station Participant
        coeffs = list(np.ones(10))
        solmag = ipfwrap.IpfReader("share/test/solmag.ipf", 8)
        zWetPD = MyScalarTimeEval()
        zDryPD = MyScalarTimeEval()
        wetScale = MyScalarTimeEval()
        dryScale = MyScalarTimeEval()
        ionoScale = MyScalarTimeEval()
        mediaCorr = obs.MediaCorrections(
            uni.frames, "New_Norcia", "New_Norcia", "Sun", "Earth", "ITRF", "ICRF", solmag, coeffs)
        mediaCorr.setWetZenithDelay(zWetPD)
        mediaCorr.setDryZenithDelay(zDryPD)
        mediaCorr.setCorrections(wetScale, dryScale, ionoScale)
        mediaCorr.setContributions(True, True)
        mediaCorr1 = obs.MediaCorrections(
            uni.frames, "Malargue", "Malargue", "Sun", "Earth", "ITRF", "ICRF", solmag, coeffs)
        mediaCorr1.setWetZenithDelay(zWetPD)
        mediaCorr1.setDryZenithDelay(zDryPD)
        mediaCorr1.setCorrections(wetScale, dryScale, ionoScale)
        mediaCorr1.setContributions(False, False)
        station_file = "share/test/GroundStationsDatabase.json"
        station_config = cosmos.util.load_json(station_file)
        station_book = station.StationBook(station_config, uni.constants)
        nn = station_book.get("New_Norcia")
        tdbtt = obs.TDBTT(eph_file="share/test/de432.jpl")
        gsClock = obs.GroundStationClock(
            uni.frames, "New_Norcia", "Earth", "ICRF", tdbtt)
        gsPar = obs.GroundStationParticipant(
            "New_Norcia", "New_Norcia", "New_Norcia", nn, gsClock, mediaCorr)
        gsPar.setTransmitterFrequency(freq)
        gsPar.setDopplerBiases({obs.FrequencyBand.X: upBias}, {
            obs.FrequencyBand.X: dnBias})
        gsPar.setRangeBiases({obs.FrequencyBand.X: upBias}, {
                             obs.FrequencyBand.X: dnBias})
        gsPar.setGroupDelays({obs.FrequencyBand.X: upDelay}, {
                             obs.FrequencyBand.X: dnDelay})
        gsPar.setPhaseDelays({obs.FrequencyBand.X: upDelay}, {
                             obs.FrequencyBand.X: dnDelay})
        mm = station_book.get("Malargue")
        gsClock1 = obs.GroundStationClock(
            uni.frames, "Malargue", "Earth", "ICRF", tdbtt)
        gsPar1 = obs.GroundStationParticipant(
            "Malargue", "Malargue", "Malargue", mm, gsClock1, mediaCorr1)
        gsPar1.setTransmitterFrequency(freq)
        gsPar1.setDopplerBiases({obs.FrequencyBand.X: upBias}, {
            obs.FrequencyBand.X: dnBias})
        gsPar1.setRangeBiases({obs.FrequencyBand.X: upBias}, {
            obs.FrequencyBand.X: dnBias})
        gsPar1.setGroupDelays({obs.FrequencyBand.X: upDelay}, {
                             obs.FrequencyBand.X: dnDelay})
        gsPar1.setPhaseDelays({obs.FrequencyBand.X: upDelay}, {
                             obs.FrequencyBand.X: dnDelay})
        # Spacecraft Participant
        groupDelay = test_param  # or common.ConstantScalarTimeEvaluable(0.)
        phaseDelay = test_param1  # or common.ConstantScalarTimeEvaluable(0.)
        trData = obs.TransponderData(
            groupDelay=groupDelay, phaseDelay=phaseDelay)
        scPar = obs.SpacecraftParticipant("Moon", {"sc_antenna": "antenna#1"}, {
                                          "sc_transponder": {"X_X": trData}}, freq)
        scPar.setTransmitterFrequency(freq)
        tau = obs.ProperTimeEquation(
            uni.frames, "Moon", "SSB", "ICRF", 0.0, [sun_bd])
        scPar.setProperTimeEquation(tau)
        # Range 1
        range1way = obs.Range1(
            scPar, gsPar, owlt, obs.FrequencyBand.X, "", False, 0.0)
        self.assertEqual(range1way.obs(epoch), range1way.compute(epoch).obs)
        self.assertEqual(range1way.obs(xepoch).value(),
                         range1way.compute(xepoch).obs.value())
        self.assertEqual(range1way.compute(epoch).delay, 1.0)
        self.assertNotEqual(range1way.compute(epoch).mediacorr, 0.0)

        # Range 2
        range2way = obs.Range2(gsPar, scPar, gsPar1, owlt, obs.BandCombination.fromString(
            "X_X"), "", "", "sc_transponder", 0.0, False, True, 0.0, 0.0)
        timetag = xepoch+86400
        result = range2way.compute(timetag)
        self.assertEqual(range2way.obs(epoch), range2way.compute(epoch).obs)
        self.assertEqual(range2way.obs(xepoch).value(),
                         range2way.compute(xepoch).obs.value())
        self.assertEqual(result.frequency1, freq.eval(timetag))
        self.assertNotEqual(result.mediacorr1, 0.0)
        self.assertEqual(result.mediacorr3, 0.0)
        self.assertEqual(result.frequency3, freq.eval(
            timetag)*range2way.transponderRatio())
        self.assertAlmostEqual(range2way.getElevations(result)[0], astro.sphericalFromCart(
            uni.frames.vector3("New_Norcia", "Moon", "New_Norcia", timetag.value()-result.lighttime12-result.lighttime23))[2], 4)
        self.assertAlmostEqual(range2way.getElevations(result)[1], astro.sphericalFromCart(
            uni.frames.vector3("Malargue", "Moon", "Malargue", timetag.value()))[2], 4)
        self.assertEqual(range2way.compute(epoch).bias1, 1.)
        self.assertEqual(range2way.compute(epoch).bias3, 0.)
        self.assertEqual(range2way.compute(epoch).delay1, 0.)
        self.assertEqual(range2way.compute(epoch).delay2, 1.)
        self.assertEqual(range2way.compute(epoch).delay3, 1.)

        # Doppler 1
        doppler1way = obs.Doppler1(
            scPar, gsPar, owlt, obs.FrequencyBand.X, 60., "", 0.0)
        self.assertEqual(doppler1way.obs(epoch),
                         doppler1way.compute(epoch).obs)
        self.assertEqual(doppler1way.obs(xepoch).value(),
                         doppler1way.compute(xepoch).obs.value())
        self.assertEqual(doppler1way.getRange1().obs(
            xepoch).value(), doppler1way.compute(xepoch).rng_start.obs.value())
        self.assertNotEqual(doppler1way.compute(epoch).delta, 0.0)

        # Doppler 2
        # With freq as ScalarTimeIntegrable
        gsPar.setTransmitterFrequency(freq1)
        doppler2way = obs.Doppler2(gsPar, scPar, gsPar, owlt, obs.BandCombination.fromString(
            "X_X"), 60., "", "", "sc_transponder", 0.0, 0.0)
        self.assertEqual(doppler2way.obs(epoch),
                         doppler2way.compute(epoch).obs)
        self.assertEqual(doppler2way.obs(xepoch).value(),
                         doppler2way.compute(xepoch).obs.value())
        self.assertEqual(doppler2way.getRange2().obs(
            xepoch).value(), doppler2way.compute(xepoch).rng_start.obs.value())
        self.assertEqual(doppler2way.compute(epoch).bias1, 1.)
        self.assertEqual(doppler2way.compute(epoch).bias3, 0.)
        # With freq as ConstantScalarTimeEvaluable
        gsPar.setTransmitterFrequency(freq)
        doppler2way = obs.Doppler2(gsPar, scPar, gsPar, owlt, obs.BandCombination.fromString(
            "X_X"), 60., "", "", "sc_transponder", 0.0, 0.0)
        self.assertEqual(doppler2way.obs(epoch),
                         doppler2way.compute(epoch).obs)
        self.assertEqual(doppler2way.obs(xepoch).value(),
                         doppler2way.compute(xepoch).obs.value())
        self.assertEqual(doppler2way.getRange2().obs(
            xepoch).value(), doppler2way.compute(xepoch).rng_start.obs.value())
        self.assertEqual(doppler2way.compute(epoch).bias1, 1.)
        self.assertEqual(doppler2way.compute(epoch).bias3, 0.)
        self.assertEqual(doppler2way.compute(epoch).rng_start.delay1, 0.)
        self.assertEqual(doppler2way.compute(epoch).rng_start.delay2, 2.)
        self.assertEqual(doppler2way.compute(epoch).rng_start.delay3, 1.)
        self.assertEqual(doppler2way.compute(epoch).rng_end.delay1, 0.)
        self.assertEqual(doppler2way.compute(epoch).rng_end.delay2, 2.)
        self.assertEqual(doppler2way.compute(epoch).rng_end.delay3, 1.)

        # SpacecraftDOR
        scDOR = obs.SpacecraftDOR(
            gsPar1, gsPar, scPar, owlt, obs.FrequencyBand.X, 8e9, 0.0, "", 0.0, 0.0)
        self.assertEqual(scDOR.obs(epoch),
                         scDOR.compute(epoch).obs)
        self.assertEqual(scDOR.obs(xepoch).value(),
                         scDOR.compute(xepoch).obs.value())

        # QuasarDOR
        qDOR = obs.QuasarDOR(gsPar, gsPar1, "Q0433", uni,
                             owlt, obs.FrequencyBand.X, 8e9, 0.0, 0.0, 0.0)
        self.assertEqual(qDOR.obs(epoch),
                         qDOR.compute(epoch).obs)
        self.assertEqual(qDOR.obs(xepoch).value(),
                         qDOR.compute(xepoch).obs.value())


if __name__ == '__main__':
    unittest.main()
