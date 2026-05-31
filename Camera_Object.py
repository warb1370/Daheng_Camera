import cv2
import sys
sys.path.append("..")
try:
    from . import gxipy as gx
except:
    import gxipy as gx
import os
import threading
import queue
import datetime
import keyboard  # pip install keyboard
import time

class Camera(object):

    def __init__(
        self,
        sn_number,
        root_video_path,
        frame_rate=30,
        save_video=True,
        show_video=True,
        project_name=None,
        stop_event=None,
        duration=None,
        # 新增：队列与策略（默认为 2 秒缓存与丢最旧，便于统计）
        queue_maxsize=180,
        drop_policy="drop_oldest",   # drop_oldest | drop_newest | block
        reset_before_config=False,
        reset_wait_timeout=10.0,
        stream_buffer_count=64,
        get_image_timeout=1000,
        gev_heartbeat_timeout=30000,
        stream_reopen_attempts=1,
        open_retries=3,
        open_retry_delay=1.0,
        empty_recovery_threshold=30
    ):
        self.project_name = project_name
        self.root_video_path = root_video_path
        self.file_name = self.get_time()[:10]
        self.save_video_path = os.path.join(self.root_video_path, self.file_name)
        self.file_exist(self.save_video_path)
        self.project_save_path = self.project_exit()
        self.sn_number = sn_number
        self.show_video = show_video
        self.stop_event = stop_event
        self.duration = duration
        self.save_video = save_video
        self.reset_before_config = reset_before_config
        self.reset_wait_timeout = reset_wait_timeout
        self.stream_buffer_count = stream_buffer_count
        self.get_image_timeout = get_image_timeout
        self.gev_heartbeat_timeout = gev_heartbeat_timeout
        self.stream_reopen_attempts = stream_reopen_attempts
        self.open_retries = open_retries
        self.open_retry_delay = open_retry_delay
        self.empty_recovery_threshold = empty_recovery_threshold
        self.offline_event = threading.Event()
        self.offline_reason = None
        self.offline_count = 0
        self._offline_callback_func = None
        self.get_image_errors = 0
        self.consecutive_empty_images = 0
        self.max_consecutive_empty_images = 0
        self.empty_image_log_every = 5
        self.last_delivered_count = None
        self.stalled_empty_images = 0
        self.stream_recoveries = 0
        self.final_stream_counters = None

        # 相机初始化
        self.device_manager = gx.DeviceManager()
        self.device_info = self._wait_for_device(self.reset_wait_timeout)
        if self.reset_before_config:
            self._reset_device_before_config()
            self.device_info = self._wait_for_device(self.reset_wait_timeout)
        self.cam = self._open_camera_with_retry()
        self.data_stream = self.cam.data_stream[0]
        self.stream_start_retries = 3
        self.stream_start_retry_delay = 0.3
        self._configure_stream_buffer()
        self._register_offline_callback()
        self._configure_transport_stability()

        # ========== AOI 设置：500x1524，按范围/步进对齐并强制偶数，Offset 尽量置 0 ==========
        def _align_value(val, rng):
            v = max(rng["min"], min(rng["max"], int(val)))
            inc = max(1, int(rng["inc"]))
            return rng["min"] + ((v - rng["min"]) // inc) * inc

        try:
            w_rng = self.cam.Width.get_range()
            h_rng = self.cam.Height.get_range()
        except Exception as e:
            w_rng = {"min": 2, "max": 99998, "inc": 2}
            h_rng = {"min": 2, "max": 99998, "inc": 2}
            print(f"[{self.sn_number}] 读取 Width/Height 范围失败，使用保守对齐：{e}")

        tgt_w = _align_value(2048, w_rng)
        tgt_h = _align_value(1536, h_rng)
        if tgt_w % 2 != 0:
            tgt_w = max(w_rng["min"], tgt_w - 1)
        if tgt_h % 2 != 0:
            tgt_h = max(h_rng["min"], tgt_h - 1)

        try:
            ox_rng = self.cam.OffsetX.get_range()
            oy_rng = self.cam.OffsetY.get_range()
        except Exception:
            ox_rng = {"min": 0, "max": 0, "inc": 1}
            oy_rng = {"min": 0, "max": 0, "inc": 1}

        try:
            if ox_rng["max"] > 0:
                self.cam.OffsetX.set(_align_value(ox_rng["min"], ox_rng))
            else:
                self.cam.OffsetX.set(0)
        except Exception as e:
            print(f"[{self.sn_number}] 设置 OffsetX 失败: {e}")

        try:
            if oy_rng["max"] > 0:
                self.cam.OffsetY.set(_align_value(oy_rng["min"], oy_rng))
            else:
                self.cam.OffsetY.set(0)
        except Exception as e:
            print(f"[{self.sn_number}] 设置 OffsetY 失败: {e}")

        try:
            self.cam.Width.set(tgt_w)
            self.cam.Height.set(tgt_h)
            print(f"[{self.sn_number}] AOI 设置完成: {tgt_w}x{tgt_h}")
        except Exception as e:
            print(f"[{self.sn_number}] 设置 AOI 失败: {e}")
        # ========== AOI 设置结束 ==========

        # 参数设置（先设置，再开流）
        try:
            self.cam.ExposureTime.set(10000)
        except Exception as e:
            print(f"[{self.sn_number}] ExposureTime.set(8000) 失败：{e}")

        try:
            self.cam.GainAuto.set(True)
        except Exception as e:
            print(f"[{self.sn_number}] GainAuto.set(True) 失败：{e}")

        self._set_enum_by_name("AcquisitionMode", ("CONTINUOUS", "Continuous"))
        self._set_enum_by_name("TriggerMode", ("OFF", "Off"))

        if self.cam.BalanceWhiteAuto.is_writable():
            try:
                self.cam.BalanceWhiteAuto.set(True)
            except Exception as e:
                print(f"[{self.sn_number}] BalanceWhiteAuto.set(True) 失败：{e}")
        else:
            print("BalanceWhiteAuto.set: is not writeable")

        if self.cam.GammaEnable.is_writable():
            try:
                self.cam.GammaEnable.set(True)
            except Exception as e:
                print(f"[{self.sn_number}] GammaEnable.set(True) 失败：{e}")
        else:
            print("GammaEnable.set: is not writeable")

        try:
            self.cam.AcquisitionFrameRateMode.set(True)
            self.cam.AcquisitionFrameRate.set(frame_rate)
        except Exception as e:
            print(f"[{self.sn_number}] 设置帧率 {frame_rate} 失败：{e}")

        # 路径与文件
        self.video_name = str(self.sn_number) + ".mp4"
        self.txt_name = str(self.sn_number) + ".txt"
        self.txt_path = os.path.join(self.project_save_path, self.txt_name)
        self.video_path = os.path.join(self.project_save_path, self.video_name)

        # 队列与线程
        self.frame_queue = queue.Queue(maxsize=queue_maxsize)
        self.drop_policy = drop_policy
        self.cut = False  # 用于写线程退出
        self.file = open(self.txt_path, "w", encoding="utf-8")

        # 清空数据流队列
        try:
            self.data_stream.flush_queue()
        except Exception as e:
            print(f"[{self.sn_number}] initial flush_queue failed: {e}")
        # 清空本地队列
        self.clear_frame_queue()
        # 开启采集流（只开一次）

        # 实测帧率与分辨率
        self.fps = int(round(self.cam.AcquisitionFrameRate.get()))
        self.size = (int(self.cam.Width.get()), int(self.cam.Height.get()))
        # 偶数兜底
        if self.size[0] % 2 != 0 or self.size[1] % 2 != 0:
            self.size = (self.size[0] - self.size[0] % 2, self.size[1] - self.size[1] % 2)
            print(f"[{self.sn_number}] 尺寸调整为偶数: {self.size}")



        # 统计
        self.stats = {
            "captured": 0,          # 采集到并完成 numpy 转换的帧数
            "enqueued": 0,          # 成功放入队列的帧数
            "dropped_queue": 0,     # 因队列满被丢弃的帧数
            "encoded": 0,           # 成功写入的帧数（VideoWriter.write 成功计数）
            "incomplete": 0,
            "convert_failed": 0,
            "queue_max_observed": 0,
            "write_waits": 0,       # block 策略阻塞次数
            "t_start": None,
            "t_end_capture": None,
            "t_end_all": None
        }

        # 写盘
        if self.save_video:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, self.fourcc, self.fps, self.size)
            self.thread = threading.Thread(target=self._write_frames, name=f"Writer-{self.sn_number}")
            self.thread.start()
            print(f"thread start for {self.sn_number}")

    def _wait_for_device(self, timeout):
        deadline = time.perf_counter() + max(0.1, float(timeout))
        last_info = None
        while time.perf_counter() < deadline:
            try:
                _, device_info_list = self.device_manager.update_all_device_list(1000)
                for info in device_info_list or []:
                    if info.get("sn") == self.sn_number:
                        return info
                last_info = device_info_list
            except Exception as e:
                last_info = e
            time.sleep(0.3)
        raise RuntimeError(f"[{self.sn_number}] device not found after waiting {timeout}s: {last_info}")

    def _open_camera_with_retry(self):
        last_error = None
        for attempt in range(1, self.open_retries + 1):
            try:
                return self.device_manager.open_device_by_sn(self.sn_number)
            except Exception as e:
                last_error = e
                print(f"[{self.sn_number}] open_device_by_sn failed ({attempt}/{self.open_retries}): {e}")
                time.sleep(self.open_retry_delay)
                try:
                    self.device_info = self._wait_for_device(self.reset_wait_timeout)
                except Exception as wait_error:
                    print(f"[{self.sn_number}] wait_for_device after open failure failed: {wait_error}")
        raise RuntimeError(f"[{self.sn_number}] open_device_by_sn failed after retries: {last_error}")

    def _reset_device_before_config(self):
        temp_cam = None
        try:
            temp_cam = self.device_manager.open_device_by_sn(self.sn_number)
            print(f"[{self.sn_number}] sending DeviceReset before parameter configuration...")
            temp_cam.DeviceReset.send_command()
        except Exception as e:
            print(f"[{self.sn_number}] DeviceReset failed or not supported: {e}")
            if self._is_gige_device() and self.device_info.get("mac"):
                try:
                    print(f"[{self.sn_number}] trying GigE reset by MAC {self.device_info.get('mac')}...")
                    self.device_manager.gige_reset_device(
                        self.device_info["mac"],
                        gx.GxResetDeviceModeEntry.RESET
                    )
                    time.sleep(1.0)
                    return True
                except Exception as reset_error:
                    print(f"[{self.sn_number}] GigE reset failed: {reset_error}")
            return False
        finally:
            if temp_cam is not None:
                try:
                    temp_cam.close_device()
                except Exception:
                    pass

        time.sleep(1.0)
        return True

    def _is_gige_device(self):
        return bool(
            self.device_info
            and self.device_info.get("device_class") == getattr(gx.GxDeviceClassList, "GEV", None)
        )

    def _configure_stream_buffer(self):
        if self.stream_buffer_count is None:
            return
        try:
            self.data_stream.set_acquisition_buffer_number(int(self.stream_buffer_count))
            print(f"[{self.sn_number}] acquisition buffer count set to {self.stream_buffer_count}")
        except Exception as e:
            print(f"[{self.sn_number}] set_acquisition_buffer_number failed: {e}")

        try:
            if self.data_stream.StreamBufferHandlingMode.is_writable():
                self.data_stream.StreamBufferHandlingMode.set(gx.GxDSStreamBufferHandlingModeEntry.OLDEST_FIRST)
        except Exception as e:
            print(f"[{self.sn_number}] StreamBufferHandlingMode set failed: {e}")

    def _configure_transport_stability(self):
        if not self._is_gige_device():
            return
        try:
            if self.cam.GevHeartbeatTimeout.is_writable():
                self.cam.GevHeartbeatTimeout.set(int(self.gev_heartbeat_timeout))
                print(f"[{self.sn_number}] GevHeartbeatTimeout set to {self.gev_heartbeat_timeout} ms")
        except Exception as e:
            print(f"[{self.sn_number}] GevHeartbeatTimeout set failed: {e}")

    def _close_camera_handle(self):
        try:
            self.cam.unregister_device_offline_callback()
        except Exception:
            pass
        try:
            if self._is_streaming():
                self.cam.stream_off()
        except Exception:
            pass
        try:
            self.cam.close_device()
        except Exception:
            pass

    def _reopen_camera_handle(self):
        print(f"[{self.sn_number}] reopening camera handle after stream_on failure...")
        self.offline_event.clear()
        self._offline_callback_func = None
        self.device_info = self._wait_for_device(self.reset_wait_timeout)
        self.cam = self._open_camera_with_retry()
        self.data_stream = self.cam.data_stream[0]
        self._configure_stream_buffer()
        self._register_offline_callback()
        self._configure_transport_stability()
        try:
            self.data_stream.flush_queue()
        except Exception as e:
            print(f"[{self.sn_number}] flush_queue after reopen failed: {e}")

    def _register_offline_callback(self):
        def on_offline():
            self._on_device_offline()

        try:
            self._offline_callback_func = on_offline
            self.cam.register_device_offline_callback(self._offline_callback_func)
            print(f"[{self.sn_number}] offline callback registered")
        except Exception as e:
            self._offline_callback_func = None
            print(f"[{self.sn_number}] offline callback unavailable: {e}")

    def _on_device_offline(self):
        self.offline_count += 1
        self.offline_reason = f"SDK offline callback at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
        self.offline_event.set()
        if self.stop_event is not None:
            self.stop_event.set()
        print(f"[{self.sn_number}] DEVICE OFFLINE: {self.offline_reason}")

    def _safe_feature_get(self, feature):
        try:
            if feature.is_readable():
                return feature.get()
        except Exception:
            return None
        return None

    def _stream_counters(self):
        counters = {
            "delivered": self._safe_feature_get(self.data_stream.StreamDeliveredFrameCount),
            "lost": self._safe_feature_get(self.data_stream.StreamLostFrameCount),
            "incomplete": self._safe_feature_get(self.data_stream.StreamIncompleteFrameCount),
            "packets": self._safe_feature_get(self.data_stream.StreamDeliveredPacketCount),
        }
        return ", ".join(f"{key}={value}" for key, value in counters.items() if value is not None) or "unavailable"

    def _delivered_count(self):
        return self._safe_feature_get(self.data_stream.StreamDeliveredFrameCount)

    def _handle_empty_image(self):
        self.consecutive_empty_images += 1
        if self.consecutive_empty_images > self.max_consecutive_empty_images:
            self.max_consecutive_empty_images = self.consecutive_empty_images
        delivered = self._delivered_count()
        if delivered is not None and delivered == self.last_delivered_count:
            self.stalled_empty_images += 1
        else:
            self.stalled_empty_images = 0
            self.last_delivered_count = delivered
        if self.consecutive_empty_images in (1, self.empty_image_log_every):
            print(
                f"[{self.sn_number}] get_image returned None "
                f"({self.consecutive_empty_images} consecutive), stream counters: {self._stream_counters()}"
            )
        elif self.consecutive_empty_images > self.empty_image_log_every:
            self.empty_image_log_every *= 2

        if self.stalled_empty_images >= self.empty_recovery_threshold:
            print(
                f"[{self.sn_number}] stream appears stalled "
                f"(delivered={delivered}, empty={self.stalled_empty_images}); recovering..."
            )
            self._recover_stalled_stream()

    def _get_image(self):
        if not self._ensure_streaming():
            raise RuntimeError(f"[{self.sn_number}] data stream is not active")
        try:
            image = self.data_stream.get_image(timeout=self.get_image_timeout)
        except Exception as e:
            self.get_image_errors += 1
            self.offline_reason = f"get_image exception: {e}"
            print(f"[{self.sn_number}] get_image error: {e}; stream counters: {self._stream_counters()}")
            raise

        if image is None:
            self._handle_empty_image()
            return None

        self.consecutive_empty_images = 0
        self.empty_image_log_every = 5
        self.stalled_empty_images = 0
        self.last_delivered_count = self._delivered_count()
        return image

    def _recover_stalled_stream(self):
        self.stream_recoveries += 1
        self.stalled_empty_images = 0
        self.consecutive_empty_images = 0
        self.empty_image_log_every = 5

        try:
            if self._is_streaming():
                self.cam.stream_off()
        except Exception as e:
            print(f"[{self.sn_number}] stream_off during recovery failed: {e}")

        try:
            self.data_stream.flush_queue()
        except Exception as e:
            print(f"[{self.sn_number}] flush_queue during recovery failed: {e}")

        self.clear_frame_queue()

        try:
            self._start_stream()
            print(f"[{self.sn_number}] stream recovery succeeded")
            self.last_delivered_count = self._delivered_count()
            return
        except Exception as e:
            print(f"[{self.sn_number}] stream recovery by restart failed: {e}")

        self._close_camera_handle()
        time.sleep(1.0)
        self._reopen_camera_handle()
        self._start_stream()
        print(f"[{self.sn_number}] stream recovery by reopen succeeded")
        self.last_delivered_count = self._delivered_count()

    def _raw_to_bgr(self, raw_image):
        try:
            status = raw_image.get_status()
        except Exception:
            status = None

        if status != gx.GxFrameStatusList.SUCCESS:
            self.stats["incomplete"] += 1
            print(f"[{self.sn_number}] skip incomplete frame, status={status}; stream counters: {self._stream_counters()}")
            return None

        rgb_image = raw_image.convert("RGB")
        if rgb_image is None:
            self.stats["convert_failed"] += 1
            print(f"[{self.sn_number}] skip frame: convert('RGB') returned None")
            return None

        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            self.stats["convert_failed"] += 1
            print(f"[{self.sn_number}] skip frame: get_numpy_array returned None")
            return None

        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    def prepare_for_recording(self):
        if not self._is_streaming():
            self._start_stream()
        try:
            self.data_stream.flush_queue()
        except Exception as e:
            print(f"[{self.sn_number}] flush_queue failed before recording: {e}")
        self.clear_frame_queue()

    def _set_enum_by_name(self, feature_name, preferred_names):
        feature = getattr(self.cam, feature_name, None)
        if feature is None:
            return False
        try:
            if not feature.is_writable():
                return False
            range_dict = feature.get_range() or {}
            options = {str(name).lower(): value for name, value in range_dict.items()}
            for name in preferred_names:
                value = options.get(str(name).lower())
                if value is not None:
                    feature.set(value)
                    return True
        except Exception as e:
            print(f"[{self.sn_number}] {feature_name} set failed: {e}")
        return False

    def _is_streaming(self):
        return bool(getattr(self.data_stream, "acquisition_flag", False))

    def _start_stream_once(self):
        last_error = None
        for attempt in range(1, self.stream_start_retries + 1):
            if self._is_streaming():
                return True
            try:
                self.cam.stream_on()
                if self._is_streaming():
                    return True
                last_error = RuntimeError("stream_on returned but acquisition_flag is still False")
            except Exception as e:
                last_error = e
                print(f"[{self.sn_number}] stream_on failed ({attempt}/{self.stream_start_retries}): {e}")
            time.sleep(self.stream_start_retry_delay)

        raise RuntimeError(f"[{self.sn_number}] stream_on failed after retries: {last_error}")

    def _start_stream(self):
        last_error = None
        for reopen_attempt in range(self.stream_reopen_attempts + 1):
            try:
                return self._start_stream_once()
            except Exception as e:
                last_error = e
                if reopen_attempt >= self.stream_reopen_attempts:
                    break
                print(f"[{self.sn_number}] stream_on still failed, reopening device and retrying: {e}")
                self._close_camera_handle()
                time.sleep(1.0)
                self._reopen_camera_handle()

        raise RuntimeError(f"[{self.sn_number}] stream_on failed after reopen attempts: {last_error}")

    def _ensure_streaming(self):
        if self._is_streaming():
            return True
        print(f"[{self.sn_number}] data stream is not active, trying to restart...")
        try:
            return self._start_stream()
        except Exception as e:
            print(f"[{self.sn_number}] data stream restart failed: {e}")
            return False

    def clear_frame_queue(self):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _write_frames(self):
        print(f"Writer thread started for {self.sn_number}")
        while True:
            item = self.frame_queue.get()
            if item is None:
                print(f"{self.sn_number}: Writer thread received stop signal, exiting.")
                break
            frame, formatted_time = item
            try:
                self.video_writer.write(frame)
                self.stats["encoded"] += 1
            except Exception as e:
                # 单帧写入失败，跳过
                print(f"[{self.sn_number}] video_writer.write error: {e}")

            # 写 timestamp
            try:
                self.file.write(f"{formatted_time}\n")
            except Exception:
                pass

            # 队列峰值统计
            qsize = self.frame_queue.qsize()
            if qsize > self.stats["queue_max_observed"]:
                self.stats["queue_max_observed"] = qsize

    def get_time(self):
        now = datetime.datetime.now()
        now_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return now_time

    def file_exist(self, file_path):
        if not os.path.exists(file_path):
            print("do not exists")
            os.makedirs(file_path, exist_ok=True)
        else:
            print("exist")

    def project_exit(self):
        if self.project_name is not None:
            path = os.path.join(self.save_video_path, self.project_name)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            return path
        else:
            self.project_name = "1"
            while os.path.exists(os.path.join(self.save_video_path, self.project_name)):
                folder_number = int(self.project_name)
                folder_number += 1
                self.project_name = str(folder_number)
            new_folder_path = os.path.join(self.save_video_path, self.project_name)
            os.makedirs(new_folder_path, exist_ok=True)
            return new_folder_path

    def get_frame(self):
        try:
            raw_image = self._get_image()
        except Exception:
            if self.stop_event is not None:
                self.stop_event.set()
            return None
        if raw_image is not None:
            formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            numpy_image = self._raw_to_bgr(raw_image)
            if numpy_image is None:
                return None

            # 采集计数
            self.stats["captured"] += 1

            if self.save_video:
                item = (numpy_image, formatted_time)
                try:
                    if self.drop_policy == "block":
                        self.frame_queue.put(item, timeout=0.5)
                        self.stats["write_waits"] += 1
                        self.stats["enqueued"] += 1
                    else:
                        self.frame_queue.put_nowait(item)
                        self.stats["enqueued"] += 1
                except queue.Full:
                    if self.drop_policy == "drop_oldest":
                        try:
                            _ = self.frame_queue.get_nowait()
                            self.stats["dropped_queue"] += 1
                            self.frame_queue.put_nowait(item)
                            self.stats["enqueued"] += 1
                        except Exception:
                            pass
                    elif self.drop_policy == "drop_newest":
                        self.stats["dropped_queue"] += 1
                    else:
                        self.stats["dropped_queue"] += 1

            return numpy_image, formatted_time
        else:
            return None

    def run(self):
        self.stats["t_start"] = time.perf_counter()
        try:
            self.prepare_for_recording()
            self.stats["t_start"] = time.perf_counter()
            while True:

                # 时长控制
                if self.duration is not None and (time.perf_counter() - self.stats["t_start"]) >= self.duration:
                    print(f"{self.sn_number}: 达到录制时长 {self.duration} 秒，自动停止。")
                    if self.stop_event is not None:
                        self.stop_event.set()
                    break

                # 外部停止
                if self.stop_event is not None and self.stop_event.is_set():
                    print(f"{self.sn_number}: stop_event detected, breaking loop")
                    break

                if self.offline_event.is_set():
                    raise RuntimeError(f"[{self.sn_number}] offline event detected: {self.offline_reason}")

                raw_image = self._get_image()
                if raw_image is None:
                    cv2.waitKey(1)
                    continue

                formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                numpy_image = self._raw_to_bgr(raw_image)
                if numpy_image is None:
                    cv2.waitKey(1)
                    continue

                # 采集计数
                self.stats["captured"] += 1

                # 显示
                if self.show_video:
                    img = cv2.resize(numpy_image, (int(self.cam.Width.get() * 0.5), int(self.cam.Height.get() * 0.5)))
                    cv2.imshow(str(self.sn_number), img)

                # 入队
                if self.save_video:
                    item = (numpy_image, formatted_time)
                    try:
                        if self.drop_policy == "block":
                            self.frame_queue.put(item, timeout=0.5)
                            self.stats["write_waits"] += 1
                            self.stats["enqueued"] += 1
                        else:
                            self.frame_queue.put_nowait(item)
                            self.stats["enqueued"] += 1
                    except queue.Full:
                        if self.drop_policy == "drop_oldest":
                            try:
                                _ = self.frame_queue.get_nowait()
                                self.stats["dropped_queue"] += 1
                                self.frame_queue.put_nowait(item)
                                self.stats["enqueued"] += 1
                            except Exception:
                                pass
                        elif self.drop_policy == "drop_newest":
                            self.stats["dropped_queue"] += 1
                        else:
                            self.stats["dropped_queue"] += 1

                # 键盘
                if self.show_video and self.sn_number == "FCM24080373":
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("Esc pressed, exiting loop")
                        if self.stop_event is not None:
                            self.stop_event.set()
                        break
                else:
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Program interrupted. Releasing resources...")
        except Exception as e:
            self.offline_reason = self.offline_reason or str(e)
            print(f"[{self.sn_number}] capture loop stopped by error: {e}")

        finally:
            # 标记采集结束时间
            self.stats["t_end_capture"] = time.perf_counter()

            print(f"Shutting down camera {self.sn_number} and cleaning up...")
            self.cut = True

            if self.save_video:
                self.frame_queue.put(None)
                self.thread.join()
                self.video_writer.release()
                print(f"Video saved to {self.video_path}")

            self.final_stream_counters = self._stream_counters()

            self._close_camera_handle()

            try:
                self.file.close()
            except Exception:
                pass

            self.stats["t_end_all"] = time.perf_counter()

            # 打印统计
            self._print_stats()
            print(f"Camera {self.sn_number} resources released.")

    def _print_stats(self):
        t_start = self.stats["t_start"] or time.perf_counter()
        t_end_cap = self.stats["t_end_capture"] or time.perf_counter()
        elapsed_cap = max(0.0, t_end_cap - t_start)

        expected = int(round(self.fps * elapsed_cap))
        captured = self.stats["captured"]
        enqueued = self.stats["enqueued"]
        encoded = self.stats["encoded"]
        dropped_queue = self.stats["dropped_queue"]

        # 丢帧定义
        drop_capture = max(0, expected - captured)
        drop_queue = dropped_queue
        drop_encode_raw = max(0, enqueued - encoded)
        drop_encode_pure = max(0, enqueued - drop_queue - encoded)

        rate_capture = (drop_capture / expected) * 100.0 if expected > 0 else 0.0
        rate_queue = (drop_queue / max(1, (drop_queue + encoded))) * 100.0
        rate_encode_pure = (drop_encode_pure / max(1, enqueued - drop_queue)) * 100.0

        print(f"[{self.sn_number}] 统计结果：")
        print(f"  目标FPS: {self.fps}")
        print(f"  录制时长(采集环节): {elapsed_cap:.3f}s")
        print(f"  期望帧数: {expected}")
        print(f"  采集帧数: {captured}")
        print(f"  入队帧数: {enqueued}")
        print(f"  队列丢帧: {dropped_queue}")
        print(f"  编码帧数: {encoded}")
        print(f"  队列峰值: {self.stats['queue_max_observed']}")
        print(f"  入队阻塞次数(block策略): {self.stats['write_waits']}")
        print(f"  采集丢帧率(相机侧，期望-采集): {rate_capture:.2f}%")
        print(f"  实时丢帧率(队列丢弃占比): {rate_queue:.2f}%")
        print(f"  纯编码丢帧数(扣除队列丢弃): {drop_encode_pure}")
        print(f"  纯编码丢帧率: {rate_encode_pure:.2f}%")
        print(f"  实际编码器: mp4v")  # 当前实现为 OpenCV VideoWriter(mp4v)
        print(f"  skipped incomplete frames: {self.stats['incomplete']}")
        print(f"  skipped conversion failures: {self.stats['convert_failed']}")
        print(f"  SDK stream counters: {self.final_stream_counters or self._stream_counters()}")
        print(f"  SDK offline callbacks: {self.offline_count}")
        print(f"  get_image exceptions: {self.get_image_errors}")
        print(f"  stream recoveries: {self.stream_recoveries}")
        print(f"  max consecutive empty images: {self.max_consecutive_empty_images}")
        if self.offline_reason:
            print(f"  stop/offline reason: {self.offline_reason}")

if __name__ == '__main__':
    project_name = "test02"
    stop_event = threading.Event()
    duration = 180

    cam1_sn = "FCM22120732"
    cam1_kwargs = {
        "root_video_path": "./Video save",
        "frame_rate": 30,
        "save_video": False,
        "show_video": True,
        "project_name": project_name,
        "stop_event": stop_event,
        "duration": duration,
        "queue_maxsize": 180,
        "drop_policy": "drop_oldest",
        "reset_before_config": True,
        "open_retries": 5,
        "open_retry_delay": 1.5,
        "stream_reopen_attempts": 2,
        "empty_recovery_threshold": 20,
    }
    # cam2 = Camera(
    #     "FCM24080373",
    #     root_video_path="./Video save",
    #     frame_rate=30,
    #     save_video=True,
    #     show_video=False,
    #     project_name=project_name,
    #     stop_event=stop_event,
    #     duration=duration,
    #     queue_maxsize=180,
    #     drop_policy="drop_oldest"
    # )

    # cam3 = Camera(
    #     "FCM23090162",
    #     root_video_path="./Video save",
    #     frame_rate=90,
    #     save_video=True,
    #     show_video=False,
    #     project_name=project_name,
    #     stop_event=stop_event,
    #     duration=duration,
    #     queue_maxsize=180,
    #     drop_policy="drop_oldest"
    # )

    t1 = None
    # t2 = threading.Thread(target=cam2.run)
    # t3 = threading.Thread(target=cam3.run)

    print("请按空格键开始录像……")
    keyboard.wait('space')
    print("检测到空格，摄像头线程启动！")
    now = time.time()
    print("空格键按下时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
          f"{now % 1:.3f}".replace("0.", "."))
    time.sleep(5)

    cam1 = Camera(cam1_sn, **cam1_kwargs)
    t1 = threading.Thread(target=cam1.run)

    t1.start()
    # t2.start()
    # t3.start()

    t1.join()
    # t2.join()
    # t3.join()

    cv2.destroyAllWindows()
    print("All cameras stopped, program finished!")
